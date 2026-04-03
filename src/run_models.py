import ast
import json
import os

import ollama
import pandas as pd
from minijinja import Environment

import load_dataset as ld

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
env = Environment(loader=lambda name: open(os.path.join(TEMPLATES_DIR, name), encoding="utf-8").read())

MODELS = [
    "mistral",
    "llama3",
    "gemma",
]


def _progress(current: int, total: int, label: str = "") -> None:
    pct = current / total * 100
    print(f"\r  [{current}/{total}] ({pct:.1f}%) {label}".ljust(80), end="", flush=True)


# ── Questões Abertas ───────────────────────────────────────────────────────────

def run_open_questions() -> None:
    df = pd.read_csv(os.path.join(ld.MY_QUESTIONS_DIR, "open_questions.csv"))
    total = len(df) * len(MODELS)
    step = 0
    results = []

    for _, row in df.iterrows():
        question = row["statement"]
        system   = row["system"]
        turns    = ast.literal_eval(row["turns"])

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": question},
        ]
        for turn in turns:
            if turn.strip():
                messages.append({"role": "user", "content": turn})

        for model in MODELS:
            step += 1
            _progress(step, total, f"q:{row['question_id']} | {model}")

            response = ollama.chat(model=model, messages=messages)
            answer   = response["message"]["content"]

            results.append({
                "question_id": row["question_id"],
                "model":       model,
                "question":    question,
                "answer":      answer,
            })

    dest = os.path.join(RESULTS_DIR, "open_questions.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\nRespostas abertas salvas em {dest}")


# ── Questões de Múltipla Escolha ───────────────────────────────────────────────

CHOICE_LABELS = ["A", "B", "C", "D"]


def run_multiple_choice_questions() -> None:
    df = pd.read_csv(os.path.join(ld.MY_QUESTIONS_DIR, "multiple_choice.csv"))
    total = len(df) * len(MODELS)
    step = 0
    results = []

    for _, row in df.iterrows():
        question = row["question"].replace("\\n", "\n")
        choices  = [
            (label, row[f"choice_{label.lower()}"].replace("\\n", "\n"))
            for label in CHOICE_LABELS
        ]

        prompt = env.render_template("multiple_choice.jinja", question=question, choices=choices)
        messages = [{"role": "user", "content": prompt}]

        for model in MODELS:
            step += 1
            _progress(step, total, f"q:{row['id']} | {model}")

            response = ollama.chat(model=model, messages=messages)
            answer   = response["message"]["content"]

            results.append({
                "question_id": row["id"],
                "model":       model,
                "question":    question,
                "choices":     dict(choices),
                "answer":      answer,
                "correct":     row["answerKey"],
            })

    dest = os.path.join(RESULTS_DIR, "multiple_choice.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\nRespostas de múltipla escolha salvas em {dest}")


# ── Execução direta ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_open_questions()
    run_multiple_choice_questions()
