from typing import Tuple, List

from datasets import load_dataset

from src.model import extract_last_int


def load_dataset_splits(cfg) -> Tuple[List[dict], List[dict]]:
    name = cfg.dataset.name.lower()
    train_split = cfg.dataset.split.train
    test_split = cfg.dataset.split.test

    if name == "gsm8k":
        train_ds = load_dataset("gsm8k", "main", split=train_split, cache_dir=".cache/")
        test_ds = load_dataset("gsm8k", "main", split=test_split, cache_dir=".cache/")
    elif name == "svamp":
        train_ds = load_dataset("svamp", split=train_split, cache_dir=".cache/")
        test_ds = load_dataset("svamp", split=test_split, cache_dir=".cache/")
    elif name in {"last_letter", "last_letter_concatenation"}:
        train_ds = load_dataset("bigbench", "last_letter_concatenation", split=train_split, cache_dir=".cache/")
        test_ds = load_dataset("bigbench", "last_letter_concatenation", split=test_split, cache_dir=".cache/")
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")

    return train_ds, test_ds


def extract_qa_pairs(dataset_name: str, dataset: List[dict]):
    name = dataset_name.lower()
    questions = []
    answers = []
    for ex in dataset:
        if name == "gsm8k":
            q = ex.get("question")
            a = ex.get("answer")
        elif name == "svamp":
            q = ex.get("Body")
            if ex.get("Question"):
                q = f"{q} {ex.get('Question')}".strip()
            a = ex.get("Answer")
        elif name in {"last_letter", "last_letter_concatenation"}:
            q = ex.get("inputs")
            a = ex.get("targets")
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        if q is None or a is None:
            raise ValueError(f"Missing fields for dataset {dataset_name}")
        questions.append(q)
        answers.append(a)
    return questions, answers


def parse_gold_answer(dataset_name: str, answer: str):
    if dataset_name.lower() in {"gsm8k", "svamp"}:
        return extract_last_int(answer)
    return str(answer).strip().lower()
