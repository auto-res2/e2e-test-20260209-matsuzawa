import os
import random
from typing import Dict, Any, List

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
import wandb

from src.preprocess import load_dataset_splits, parse_gold_answer, extract_qa_pairs
from src.model import LLMWrapper, MFTAutoCoTSelector, TriadAutoCoTSelector


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_mode_overrides(cfg) -> None:
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.dataset.split.train = "train[:40]"
        cfg.dataset.split.test = "test[:20]"
        cfg.training.epochs = 1
        cfg.training.clustering.k_clusters = min(cfg.training.clustering.k_clusters, 3)
        cfg.training.clustering.C_candidates_per_cluster = min(
            cfg.training.clustering.C_candidates_per_cluster, 2
        )
        if "mft_autocot" in cfg.training:
            cfg.training.mft_autocot.max_views = min(cfg.training.mft_autocot.max_views, 4)
            cfg.training.mft_autocot.K_paraphrases_max = min(cfg.training.mft_autocot.K_paraphrases_max, 1)
        if "triad_autocot" in cfg.training:
            cfg.training.triad_autocot.fixed_views_budget = min(cfg.training.triad_autocot.fixed_views_budget, 4)
        cfg.evaluation.paraphrase_k = min(cfg.evaluation.paraphrase_k, 2)
        cfg.training.demo_construction_max_candidates = 2
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


def _sanity_gradient_check(model_wrapper: LLMWrapper, cfg) -> None:
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    model.train()
    text = "Sanity check for gradient flow."
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    labels = inputs["input_ids"].clone()
    assert inputs["input_ids"].shape == labels.shape, "Input/label shapes must match"
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss

    grads = torch.autograd.grad(
        loss,
        [p for p in model.parameters() if p.requires_grad],
        create_graph=False,
        retain_graph=True,
        allow_unused=True,
    )
    assert any(g is not None and g.detach().abs().sum().item() > 0 for g in grads), "Aux grads are all zero"

    loss.backward()

    grad_sums = []
    for p in model.parameters():
        if p.grad is not None:
            grad_sums.append(p.grad.detach().abs().sum().item())
    assert len(grad_sums) > 0, "No gradients found before optimizer.step()"
    assert sum(grad_sums) > 0.0, "All gradients are zero before optimizer.step()"

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()


def _build_selector(cfg, model_wrapper: LLMWrapper):
    method = cfg.method.lower()
    if "mft" in method:
        return MFTAutoCoTSelector(model_wrapper, cfg)
    if "triad" in method:
        return TriadAutoCoTSelector(model_wrapper, cfg)
    raise ValueError(f"Unsupported method: {cfg.method}")


def _run_pipeline(cfg, model_wrapper: LLMWrapper, use_wandb: bool = True) -> Dict[str, Any]:
    train_ds, test_ds = load_dataset_splits(cfg)
    train_questions, train_answers = extract_qa_pairs(cfg.dataset.name, train_ds)

    selector = _build_selector(cfg, model_wrapper)
    demo_prompt, demos, selection_stats = selector.build_demos(
        train_questions,
        train_answers,
        use_wandb=use_wandb,
        max_candidates=cfg.training.get("demo_construction_max_candidates", None),
    )

    demo_correctness = []
    for d in demos:
        gold = parse_gold_answer(cfg.dataset.name, d.get("gold_answer", ""))
        demo_correctness.append(int(d["answer"] == gold))
    demo_correctness_rate = float(np.mean(demo_correctness)) if demos else 0.0

    fragilities = []
    for d in demos:
        frag = selector.compute_paraphrase_fragility(
            d["question"],
            d["answer"],
            k_paraphrase=cfg.evaluation.paraphrase_k,
        )
        fragilities.append(frag)
    paraphrase_fragility = float(np.mean(fragilities)) if fragilities else 1.0

    correct = 0
    per_question_correctness = []
    question_lengths = []
    max_test_batches = 2 if cfg.mode == "trial" else None

    for i, ex in enumerate(test_ds):
        if max_test_batches is not None and i >= max_test_batches:
            break
        question, gold_raw = extract_qa_pairs(cfg.dataset.name, [ex])
        question = question[0]
        gold = parse_gold_answer(cfg.dataset.name, gold_raw[0])
        pred = selector.answer_with_demos(demo_prompt, question)
        if i == 0:
            tok = model_wrapper.tokenizer(demo_prompt + question, return_tensors="pt")
            labels = tok["input_ids"].clone()
            assert tok["input_ids"].shape == labels.shape, "Input/label shapes must match at batch start"
        is_correct = int(pred == gold)
        correct += is_correct
        per_question_correctness.append(is_correct)
        question_lengths.append(len(question.split()))
        running_acc = correct / (i + 1)
        if use_wandb and wandb.run is not None:
            wandb.log({
                "test_correct": is_correct,
                "test_accuracy_running": running_acc,
                "test_step": i,
                "test_question_length": question_lengths[-1],
            })

    accuracy = correct / max(1, len(per_question_correctness))

    metrics = {
        "accuracy": accuracy,
        "demo_correctness_rate": demo_correctness_rate,
        "avg_llm_calls_per_accepted_demo": selection_stats["avg_calls_per_demo"],
        "rejection_rate": selection_stats["rejection_rate"],
        "paraphrase_fragility_selected_demos": paraphrase_fragility,
        "num_demos": len(demos),
        "avg_demo_score": selection_stats.get("avg_demo_score", 0.0),
        "avg_views_per_demo": selection_stats.get("avg_views_per_demo", 0.0),
        "avg_agreement_p": selection_stats.get("avg_p", 0.0),
        "avg_invariance_v": selection_stats.get("avg_v", 0.0),
        "rejection_reason_counts": selection_stats.get("rejection_reason_counts", {}),
        "total_llm_calls": selector.calls,
    }

    if use_wandb and wandb.run is not None:
        for k, v in metrics.items():
            wandb.summary[k] = v
        wandb.summary["per_question_correctness"] = per_question_correctness
        wandb.summary["test_question_lengths"] = question_lengths
        wandb.summary["selected_demos"] = demos
        wandb.summary["demo_correctness_list"] = demo_correctness

    return metrics


def _optuna_objective(trial, cfg, model_wrapper: LLMWrapper):
    for sp in cfg.optuna.search_spaces:
        name = sp.param_name
        if sp.distribution_type == "categorical":
            val = trial.suggest_categorical(name, sp.choices)
        elif sp.distribution_type == "uniform":
            val = trial.suggest_float(name, sp.low, sp.high)
        else:
            raise ValueError(f"Unsupported distribution type: {sp.distribution_type}")
        if name in ["k_clusters", "C_candidates_per_cluster"]:
            setattr(cfg.training.clustering, name, val)
        elif name in ["S_modes_max", "K_paraphrases_max", "max_views", "accept_p", "reject_p", "tau_score_threshold"]:
            setattr(cfg.training.mft_autocot, name, val)
        elif name in ["S_modes", "R_samples_per_mode", "fixed_views_budget", "accept_p"]:
            setattr(cfg.training.triad_autocot, name, val)
        elif name == "temperature_solve":
            cfg.training.generation.temperature_solve = val

    cfg.dataset.split.train = "train[:120]"
    cfg.dataset.split.test = "test[:50]"
    metrics = _run_pipeline(cfg, model_wrapper, use_wandb=False)
    return metrics["accuracy"]


@hydra.main(config_path="../config", config_name="config")
def main(cfg) -> None:
    apply_mode_overrides(cfg)
    os.makedirs(cfg.results_dir, exist_ok=True)
    set_seed(cfg.seed)

    model_wrapper = LLMWrapper(cfg)

    assert model_wrapper.tokenizer.pad_token_id is not None, "Tokenizer pad_token_id must be set"
    assert model_wrapper.model.get_output_embeddings().weight.shape[0] == model_wrapper.tokenizer.vocab_size, (
        "Model output dimension must match tokenizer vocab size"
    )

    _sanity_gradient_check(model_wrapper, cfg)

    if cfg.optuna.n_trials and cfg.optuna.n_trials > 0:
        import optuna

        study = optuna.create_study(direction="maximize")
        cfg_copy = OmegaConf.deepcopy(cfg)
        study.optimize(lambda t: _optuna_objective(t, cfg_copy, model_wrapper), n_trials=cfg.optuna.n_trials)
        best_params = study.best_params
        for name, val in best_params.items():
            if name in ["k_clusters", "C_candidates_per_cluster"]:
                setattr(cfg.training.clustering, name, val)
            elif name in ["S_modes_max", "K_paraphrases_max", "max_views", "accept_p", "reject_p", "tau_score_threshold"]:
                setattr(cfg.training.mft_autocot, name, val)
            elif name in ["S_modes", "R_samples_per_mode", "fixed_views_budget", "accept_p"]:
                setattr(cfg.training.triad_autocot, name, val)
            elif name == "temperature_solve":
                cfg.training.generation.temperature_solve = val

    use_wandb = cfg.wandb.mode != "disabled"
    if use_wandb:
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
        )

    metrics = _run_pipeline(cfg, model_wrapper, use_wandb=use_wandb)

    if use_wandb and wandb.run is not None:
        wandb.log({"accuracy": metrics["accuracy"]})
        print(wandb.run.url)
        wandb.finish()


if __name__ == "__main__":
    main()
