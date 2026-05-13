import ast
import json
import os
import re

import logging

import ollama
import pandas as pd
from minijinja import Environment

import load_dataset as ld

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
env = Environment(loader=lambda name: open(os.path.join(TEMPLATES_DIR, name), encoding="utf-8").read())

MODELS = [
    "llama3.2:3b",
    "gemma2:2b",
    "qwen2.5:3b",
]

OLLAMA_TIMEOUT = 120.0
client = ollama.Client(timeout=OLLAMA_TIMEOUT)


def _progress(current: int, total: int, label: str = "") -> None:
    pct = current / total * 100
    print(f"\r  [{current}/{total}] ({pct:.1f}%) {label}".ljust(80), end="", flush=True)


# ── Questões Abertas ───────────────────────────────────────────────────────────

def run_open_questions() -> None:
    df = pd.read_csv(os.path.join(ld.MY_QUESTIONS_DIR, "open_questions.csv"))
    dest = os.path.join(RESULTS_DIR, "open_questions.json")

    if os.path.exists(dest):
        with open(dest, encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []
    done = {(r["question_id"], r["model"]) for r in results}

    total = sum(1 for _, row in df.iterrows() for m in MODELS if (row["question_id"], m) not in done)
    step = 0
    if total == 0:
        print(f"Nada a fazer — {len(results)} entradas já presentes em {dest}.")
        return

    for _, row in df.iterrows():
        pending_models = [m for m in MODELS if (row["question_id"], m) not in done]
        if not pending_models:
            continue

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

        for model in pending_models:
            step += 1
            _progress(step, total, f"q:{row['question_id']} | {model}")

            try:
                response = client.chat(model=model, messages=messages)
                answer   = response["message"]["content"]
            except Exception as e:
                logger.warning("Ollama falhou em q:%s model:%s — %s", row["question_id"], model, e)
                answer = ""

            results.append({
                "question_id": row["question_id"],
                "model":       model,
                "question":    question,
                "answer":      answer,
            })
            done.add((row["question_id"], model))

    with open(dest, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\nRespostas abertas salvas em {dest}")


# ── Questões de Múltipla Escolha ───────────────────────────────────────────────

CHOICE_LABELS = ["A", "B", "C", "D"]


def run_multiple_choice_questions() -> None:
    df = pd.read_csv(os.path.join(ld.MY_QUESTIONS_DIR, "multiple_choice.csv"))
    dest = os.path.join(RESULTS_DIR, "multiple_choice.json")

    if os.path.exists(dest):
        with open(dest, encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []
    done = {(r["question_id"], r["model"]) for r in results}

    total = sum(1 for _, row in df.iterrows() for m in MODELS if (row["id"], m) not in done)
    step = 0
    if total == 0:
        print(f"Nada a fazer — {len(results)} entradas já presentes em {dest}.")
        return

    for _, row in df.iterrows():
        pending_models = [m for m in MODELS if (row["id"], m) not in done]
        if not pending_models:
            continue

        question = row["question"].replace("\\n", "\n")
        choices  = [
            (label, row[f"choice_{label.lower()}"].replace("\\n", "\n"))
            for label in CHOICE_LABELS
        ]

        system_prompt = env.render_template("multiple_choice_system.jinja")
        user_prompt = env.render_template("multiple_choice.jinja", question=question, choices=choices)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for model in pending_models:
            step += 1
            _progress(step, total, f"q:{row['id']} | {model}")

            try:
                response = client.chat(model=model, messages=messages)
                answer   = response["message"]["content"]
            except Exception as e:
                logger.warning("Ollama falhou em q:%s model:%s — %s", row["id"], model, e)
                answer = ""

            results.append({
                "question_id": row["id"],
                "model":       model,
                "question":    question,
                "choices":     dict(choices),
                "answer":      answer,
                "correct":     row["answerKey"],
            })
            done.add((row["id"], model))

    with open(dest, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\nRespostas de múltipla escolha salvas em {dest}")


# ── Curadoria (dificuldade + legislação) ──────────────────────────────────────

CURATOR_MODEL = "llama3.2:3b"


def _extract_json(text: str) -> str | None:
    text = text.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None


def run_curator_tasks() -> None:
    """Classifica dificuldade e identifica legislação base de cada questão."""
    df_open = pd.read_csv(os.path.join(ld.MY_QUESTIONS_DIR, "open_questions.csv"))
    df_mc = pd.read_csv(os.path.join(ld.MY_QUESTIONS_DIR, "multiple_choice.csv"))
    dest = os.path.join(RESULTS_DIR, "curator_annotations.json")

    if os.path.exists(dest):
        with open(dest, encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []
    done = {r["question_id"] for r in results}

    all_questions = []

    for _, row in df_open.iterrows():
        if row["question_id"] in done:
            continue
        turns = ast.literal_eval(row["turns"])
        all_questions.append({
            "question_id": row["question_id"],
            "statement": row["statement"],
            "turns": turns,
            "type": "open",
        })

    for _, row in df_mc.iterrows():
        if row["id"] in done:
            continue
        all_questions.append({
            "question_id": row["id"],
            "statement": row["question"].replace("\\n", "\n"),
            "turns": None,
            "type": "multiple_choice",
        })

    total = len(all_questions)
    if total == 0:
        print(f"Nada a fazer — {len(results)} anotações já presentes em {dest}.")
        return

    for i, q in enumerate(all_questions, 1):
        _progress(i, total, f"curadoria: {q['question_id']}")

        difficulty_prompt = env.render_template(
            "curator_difficulty.jinja",
            question_id=q["question_id"],
            statement=q["statement"],
            turns=q["turns"],
        )
        try:
            diff_resp = client.chat(
                model=CURATOR_MODEL,
                options={"temperature": 0},
                messages=[{"role": "user", "content": difficulty_prompt}],
            )
        except Exception as e:
            logger.warning("Ollama falhou (dificuldade) q:%s — %s", q["question_id"], e)
            diff_resp = None

        legislation_prompt = env.render_template(
            "curator_legislation.jinja",
            question_id=q["question_id"],
            statement=q["statement"],
            turns=q["turns"],
        )
        try:
            leg_resp = client.chat(
                model=CURATOR_MODEL,
                options={"temperature": 0},
                messages=[{"role": "user", "content": legislation_prompt}],
            )
        except Exception as e:
            logger.warning("Ollama falhou (legislação) q:%s — %s", q["question_id"], e)
            leg_resp = None

        specialty_prompt = env.render_template(
            "curator_specialty.jinja",
            question_id=q["question_id"],
            statement=q["statement"],
            turns=q["turns"],
        )
        try:
            spec_resp = client.chat(
                model=CURATOR_MODEL,
                options={"temperature": 0},
                messages=[{"role": "user", "content": specialty_prompt}],
            )
        except Exception as e:
            logger.warning("Ollama falhou (especialidade) q:%s — %s", q["question_id"], e)
            spec_resp = None

        entry = {"question_id": q["question_id"], "type": q["type"]}

        try:
            diff_json = json.loads(_extract_json(diff_resp["message"]["content"]))
            entry["dificuldade"] = diff_json.get("dificuldade")
            entry["nivel"] = diff_json.get("nivel", "")
        except Exception as e:
            logger.warning("JSON inválido (dificuldade) q:%s — %s", q["question_id"], e)
            entry["dificuldade"] = None
            entry["nivel"] = None

        try:
            leg_json = json.loads(_extract_json(leg_resp["message"]["content"]))
            entry["corpus_referencia"] = leg_json.get("corpus_referencia", "")
        except Exception as e:
            logger.warning("JSON inválido (corpus de referência) q:%s — %s", q["question_id"], e)
            entry["corpus_referencia"] = None

        try:
            spec_json = json.loads(_extract_json(spec_resp["message"]["content"]))
            entry["subdominio_semantico"] = spec_json.get("subdominio_semantico", "")
        except Exception as e:
            logger.warning("JSON inválido (subdomínio semântico) q:%s — %s", q["question_id"], e)
            entry["subdominio_semantico"] = None

        results.append(entry)
        done.add(q["question_id"])

    with open(dest, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\nAnotações de curadoria salvas em {dest}")


# ── Execução direta ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_open_questions()
    run_multiple_choice_questions()
    run_curator_tasks()
