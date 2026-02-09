import re
from collections import Counter
from typing import Dict, List, Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb


THINK_MODES = {
    "direct": "Let's think step by step.",
    "algebra": "Solve using equations/algebra. Show steps.",
    "sanity": "Solve and include a quick estimation/sanity-check.",
    "constraints": "Solve by checking constraints/units and consistency at each step.",
}


def extract_last_int(text: str):
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None


def extract_last_token(text: str):
    toks = re.findall(r"[A-Za-z]+", text)
    return toks[-1].lower() if toks else None


def resolve_model_name(name: str) -> str:
    mapping = {
        "Qwen3-8B": "Qwen/Qwen2.5-7B-Instruct",
    }
    return mapping.get(name, name)


class LLMWrapper:
    def __init__(self, cfg):
        model_name = resolve_model_name(cfg.model.name)
        torch_dtype = torch.bfloat16 if cfg.model.precision == "bf16" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=".cache/")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=".cache/",
            device_map="auto",
            torch_dtype=torch_dtype,
        )
        self.model.eval()
        self.max_tokens = cfg.model.max_tokens

    def generate(self, prompt: str, temperature: float, max_new_tokens: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]
        do_sample = temperature > 0
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen_tokens = outputs[0][input_len:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return text.strip()


class DemoSelectorBase:
    def __init__(self, model_wrapper: LLMWrapper, cfg):
        self.model_wrapper = model_wrapper
        self.cfg = cfg
        self.calls = 0

    def _chat(self, prompt: str, temperature: float) -> str:
        self.calls += 1
        return self.model_wrapper.generate(prompt, temperature, self.cfg.model.max_tokens)

    def solve(self, question: str, mode_key: str, temperature: float) -> str:
        prompt = f"Q: {question}\nA: {THINK_MODES[mode_key]}"
        return self._chat(prompt, temperature)

    def paraphrase(self, question: str, temperature: float) -> str:
        prompt = (
            "Paraphrase the following problem WITHOUT changing its meaning. "
            "Keep all numbers/units identical. Output only the paraphrased problem.\n\n"
            f"Problem: {question}"
        )
        return self._chat(prompt, temperature).strip()

    def critique(self, question: str, rationale: str, temperature: float) -> int:
        prompt = (
            "You are a meticulous auditor. If you find any concrete mistake in the solution, "
            "output exactly 'ERROR'. Otherwise output exactly 'NO ERROR'.\n\n"
            f"Question: {question}\n\nSolution:\n{rationale}\n"
        )
        out = self._chat(prompt, temperature).strip().upper()
        return 1 if "ERROR" in out else 0

    def answer_with_demos(self, demo_prompt: str, question: str):
        prompt = demo_prompt + f"Q: {question}\nA: Let's think step by step."
        rat = self._chat(prompt, temperature=0.0)
        return extract_last_int(rat)

    def compute_paraphrase_fragility(self, question: str, original_answer: int, k_paraphrase: int) -> float:
        if k_paraphrase <= 0:
            return 0.0
        paraphrases = [
            self.paraphrase(question, self.cfg.training.generation.temperature_paraphrase)
            for _ in range(k_paraphrase)
        ]
        mismatches = 0
        for pq in paraphrases:
            pred = self.answer_with_demos("", pq)
            mismatches += int(pred != original_answer)
        return mismatches / k_paraphrase


class MFTAutoCoTSelector(DemoSelectorBase):
    def score_question(self, question: str, answer_type: str):
        cfg = self.cfg.training.mft_autocot
        mode_keys = list(THINK_MODES.keys())[: cfg.S_modes_max]
        rewrites = [question]
        answers, rationales, rewrite_ids = [], [], []
        view_budget = 0
        stage = 0
        rejection_reason = None

        def parse_ans(txt):
            return extract_last_int(txt) if answer_type == "int" else extract_last_token(txt)

        def agreement():
            vals = [a for a in answers if a is not None]
            if not vals:
                return 0.0, None
            c = Counter(vals)
            a_star, cnt = c.most_common(1)[0]
            return cnt / len(vals), a_star

        def paraphrase_invariance(a_star):
            if a_star is None:
                return 0.0
            ok, total = 0, 0
            for rid in set(rewrite_ids):
                vals = [a for a, r in zip(answers, rewrite_ids) if r == rid and a is not None]
                if not vals:
                    continue
                total += 1
                if Counter(vals).most_common(1)[0][0] == a_star:
                    ok += 1
            return ok / total if total else 0.0

        while view_budget < cfg.max_views:
            if stage > 0 and len(rewrites) < (1 + cfg.K_paraphrases_max):
                rewrites.append(self.paraphrase(question, self.cfg.training.generation.temperature_paraphrase))

            for rid, rq in enumerate(rewrites):
                for mk in mode_keys:
                    if view_budget >= cfg.max_views:
                        break
                    rat = self.solve(rq, mk, self.cfg.training.generation.temperature_solve)
                    a = parse_ans(rat)
                    answers.append(a)
                    rationales.append(rat)
                    rewrite_ids.append(rid)
                    view_budget += 1
                    if wandb.run is not None:
                        wandb.log({"view_budget": view_budget, "current_agreement": agreement()[0]})

            p, a_star = agreement()
            if cfg.adaptive_stopping and p <= cfg.reject_p and view_budget >= len(mode_keys):
                rejection_reason = "low_agreement"
                return None, view_budget, rejection_reason

            if cfg.adaptive_stopping and p >= cfg.accept_p and a_star is not None:
                reps = [r for a, r in zip(answers, rationales) if a == a_star and a is not None]
                z_star = min(reps, key=len)
                e = self.critique(question, z_star, self.cfg.training.generation.temperature_audit)
                if e == 1:
                    rejection_reason = "audit_fail"
                    return None, view_budget, rejection_reason
                v = paraphrase_invariance(a_star)
                g = 1.0 if parse_ans(z_star) is not None else 0.0
                score = (p * max(v, 1e-6)) * (1 - e) * g
                if score < cfg.tau_score_threshold:
                    rejection_reason = "low_score"
                    return None, view_budget, rejection_reason
                return {
                    "score": score,
                    "question": question,
                    "rationale": z_star,
                    "answer": a_star,
                    "p": p,
                    "v": v,
                    "audit_error": e,
                    "views": view_budget,
                }, view_budget, None

            stage += 1
            if len(mode_keys) < cfg.S_modes_max:
                mode_keys = list(THINK_MODES.keys())[: min(cfg.S_modes_max, len(mode_keys) + 1)]
            if len(rewrites) >= (1 + cfg.K_paraphrases_max) and len(mode_keys) >= cfg.S_modes_max:
                break

        p, a_star = agreement()
        if a_star is None:
            rejection_reason = "no_answer"
            return None, view_budget, rejection_reason
        reps = [r for a, r in zip(answers, rationales) if a == a_star and a is not None]
        z_star = min(reps, key=len)
        e = self.critique(question, z_star, self.cfg.training.generation.temperature_audit)
        if e == 1:
            rejection_reason = "audit_fail"
            return None, view_budget, rejection_reason
        v = paraphrase_invariance(a_star)
        g = 1.0 if parse_ans(z_star) is not None else 0.0
        score = (p * max(v, 1e-6)) * (1 - e) * g
        if score < cfg.tau_score_threshold:
            rejection_reason = "low_score"
            return None, view_budget, rejection_reason
        return {
            "score": score,
            "question": question,
            "rationale": z_star,
            "answer": a_star,
            "p": p,
            "v": v,
            "audit_error": e,
            "views": view_budget,
        }, view_budget, None

    def build_demos(self, questions: List[str], answers: List[str], use_wandb: bool = True, max_candidates: int = None):
        emb_model = SentenceTransformer(self.cfg.training.clustering.embed_model, cache_folder=".cache/")
        emb = emb_model.encode(questions, normalize_embeddings=True)
        km = KMeans(
            n_clusters=self.cfg.training.clustering.k_clusters,
            random_state=0,
            n_init="auto",
        ).fit(emb)

        demos = []
        total_candidates = 0
        rejected = 0
        rejection_counts = Counter()
        accepted_calls = []
        processed_candidates = 0

        for c in range(self.cfg.training.clustering.k_clusters):
            idxs = np.where(km.labels_ == c)[0]
            if len(idxs) == 0:
                continue
            center = km.cluster_centers_[c]
            dists = np.linalg.norm(emb[idxs] - center, axis=1)
            cand = idxs[np.argsort(dists)[: min(self.cfg.training.clustering.C_candidates_per_cluster, len(idxs))]]
            total_candidates += len(cand)

            best = None
            for i in cand:
                if max_candidates is not None and processed_candidates >= max_candidates:
                    break
                processed_candidates += 1
                before_calls = self.calls
                res, views, rejection_reason = self.score_question(questions[i], self.cfg.dataset.preprocessing.answer_type)
                calls_used = self.calls - before_calls
                if res is None:
                    rejected += 1
                    if rejection_reason:
                        rejection_counts[rejection_reason] += 1
                    if use_wandb and wandb.run is not None:
                        wandb.log({
                            "candidate_rejected": 1,
                            "candidate_calls_used": calls_used,
                            "rejection_reason": rejection_reason or "unknown",
                        })
                    continue
                res["gold_answer"] = answers[i]
                res["calls_used"] = calls_used
                res["cluster_id"] = c
                accepted_calls.append(calls_used)
                if use_wandb and wandb.run is not None:
                    wandb.log({
                        "candidate_rejected": 0,
                        "candidate_calls_used": calls_used,
                        "candidate_score": res["score"],
                        "candidate_p": res["p"],
                        "candidate_v": res["v"],
                    })
                if best is None or res["score"] > best["score"]:
                    best = res

            if best is not None:
                demos.append(best)
                if use_wandb and wandb.run is not None:
                    wandb.log({
                        "cluster_id": c,
                        "cluster_best_score": best["score"],
                        "cluster_best_p": best["p"],
                        "cluster_best_v": best["v"],
                        "cluster_best_views": best["views"],
                    })

        demo_prompt = "".join([f"Q: {d['question']}\nA: {d['rationale']}\n\n" for d in demos])
        stats = {
            "avg_calls_per_demo": float(np.mean(accepted_calls)) if accepted_calls else 0.0,
            "rejection_rate": rejected / max(1, total_candidates),
            "avg_demo_score": float(np.mean([d["score"] for d in demos])) if demos else 0.0,
            "avg_views_per_demo": float(np.mean([d["views"] for d in demos])) if demos else 0.0,
            "avg_p": float(np.mean([d["p"] for d in demos])) if demos else 0.0,
            "avg_v": float(np.mean([d["v"] for d in demos])) if demos else 0.0,
            "rejection_reason_counts": dict(rejection_counts),
        }
        return demo_prompt, demos, stats


class TriadAutoCoTSelector(DemoSelectorBase):
    def score_question(self, question: str, answer_type: str):
        cfg = self.cfg.training.triad_autocot
        mode_keys = list(THINK_MODES.keys())[: cfg.S_modes]
        answers, rationales = [], []
        view_budget = 0

        def parse_ans(txt):
            return extract_last_int(txt) if answer_type == "int" else extract_last_token(txt)

        while view_budget < cfg.fixed_views_budget:
            for mk in mode_keys:
                if view_budget >= cfg.fixed_views_budget:
                    break
                for _ in range(cfg.R_samples_per_mode):
                    if view_budget >= cfg.fixed_views_budget:
                        break
                    rat = self.solve(question, mk, self.cfg.training.generation.temperature_solve)
                    a = parse_ans(rat)
                    answers.append(a)
                    rationales.append(rat)
                    view_budget += 1
                    if wandb.run is not None:
                        wandb.log({"view_budget": view_budget})

        vals = [a for a in answers if a is not None]
        if not vals:
            return None, view_budget, "no_answer"
        c = Counter(vals)
        a_star, cnt = c.most_common(1)[0]
        p = cnt / len(vals)
        if p < cfg.accept_p:
            return None, view_budget, "low_agreement"
        reps = [r for a, r in zip(answers, rationales) if a == a_star and a is not None]
        z_star = min(reps, key=len)
        e = self.critique(question, z_star, self.cfg.training.generation.temperature_audit)
        if e == 1:
            return None, view_budget, "audit_fail"
        score = p * (1 - e)
        return {
            "score": score,
            "question": question,
            "rationale": z_star,
            "answer": a_star,
            "p": p,
            "v": 0.0,
            "audit_error": e,
            "views": view_budget,
        }, view_budget, None

    def build_demos(self, questions: List[str], answers: List[str], use_wandb: bool = True, max_candidates: int = None):
        emb_model = SentenceTransformer(self.cfg.training.clustering.embed_model, cache_folder=".cache/")
        emb = emb_model.encode(questions, normalize_embeddings=True)
        km = KMeans(
            n_clusters=self.cfg.training.clustering.k_clusters,
            random_state=0,
            n_init="auto",
        ).fit(emb)

        demos = []
        total_candidates = 0
        rejected = 0
        rejection_counts = Counter()
        accepted_calls = []
        processed_candidates = 0

        for c in range(self.cfg.training.clustering.k_clusters):
            idxs = np.where(km.labels_ == c)[0]
            if len(idxs) == 0:
                continue
            center = km.cluster_centers_[c]
            dists = np.linalg.norm(emb[idxs] - center, axis=1)
            cand = idxs[np.argsort(dists)[: min(self.cfg.training.clustering.C_candidates_per_cluster, len(idxs))]]
            total_candidates += len(cand)

            best = None
            for i in cand:
                if max_candidates is not None and processed_candidates >= max_candidates:
                    break
                processed_candidates += 1
                before_calls = self.calls
                res, views, rejection_reason = self.score_question(questions[i], self.cfg.dataset.preprocessing.answer_type)
                calls_used = self.calls - before_calls
                if res is None:
                    rejected += 1
                    if rejection_reason:
                        rejection_counts[rejection_reason] += 1
                    if use_wandb and wandb.run is not None:
                        wandb.log({
                            "candidate_rejected": 1,
                            "candidate_calls_used": calls_used,
                            "rejection_reason": rejection_reason or "unknown",
                        })
                    continue
                res["gold_answer"] = answers[i]
                res["calls_used"] = calls_used
                res["cluster_id"] = c
                accepted_calls.append(calls_used)
                if use_wandb and wandb.run is not None:
                    wandb.log({
                        "candidate_rejected": 0,
                        "candidate_calls_used": calls_used,
                        "candidate_score": res["score"],
                        "candidate_p": res["p"],
                    })
                if best is None or res["score"] > best["score"]:
                    best = res

            if best is not None:
                demos.append(best)
                if use_wandb and wandb.run is not None:
                    wandb.log({
                        "cluster_id": c,
                        "cluster_best_score": best["score"],
                        "cluster_best_p": best["p"],
                        "cluster_best_views": best["views"],
                    })

        demo_prompt = "".join([f"Q: {d['question']}\nA: {d['rationale']}\n\n" for d in demos])
        stats = {
            "avg_calls_per_demo": float(np.mean(accepted_calls)) if accepted_calls else 0.0,
            "rejection_rate": rejected / max(1, total_candidates),
            "avg_demo_score": float(np.mean([d["score"] for d in demos])) if demos else 0.0,
            "avg_views_per_demo": float(np.mean([d["views"] for d in demos])) if demos else 0.0,
            "avg_p": float(np.mean([d["p"] for d in demos])) if demos else 0.0,
            "avg_v": 0.0,
            "rejection_reason_counts": dict(rejection_counts),
        }
        return demo_prompt, demos, stats
