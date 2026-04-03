import ast
import json
import os
import re

import matplotlib.pyplot as plt
import ollama
import pandas as pd
from minijinja import Environment

import load_dataset as ld

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
env = Environment(loader=lambda name: open(os.path.join(TEMPLATES_DIR, name), encoding="utf-8").read())

JUDGE_MODEL = "llama3"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _progress(current: int, total: int, label: str = "") -> None:
    pct = current / total * 100
    print(f"\r  [{current}/{total}] ({pct:.1f}%) {label}".ljust(80), end="", flush=True)



def _extract_json(text: str) -> str | None:
    """Extrai o primeiro objeto JSON encontrado no texto."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None


def _clean_score(value) -> float:
    """Converte formatos variados de nota (ex: '0,60', '0.00/0,10') para float."""
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value)
    match = re.search(r"\d+[.,]?\d*", value)
    return float(match.group(0).replace(",", ".")) if match else 0.0


# ── Avaliação: Questões Abertas (rubrica) ──────────────────────────────────────

def evaluate_open_questions() -> pd.DataFrame:
    """Avalia questões abertas usando rubrica oficial e um modelo juiz."""
    with open(os.path.join(RESULTS_DIR, "open_questions.json"), encoding="utf-8") as f:
        answers = json.load(f)

    answers_df = pd.DataFrame(answers)
    questions_df = pd.read_csv(os.path.join(ld.OPEN_DIR, "questions.csv"))
    guidelines_df = pd.read_csv(os.path.join(ld.OPEN_DIR, "guidelines.csv"))

    total = len(answers_df)
    results = []

    for i, (_, row) in enumerate(answers_df.iterrows(), 1):
        question_id = row["question_id"]
        model = row["model"]
        answer = row["answer"]
        _progress(i, total, f"q:{question_id} | {model}")

        q = questions_df[questions_df["question_id"] == question_id].iloc[0]
        g = guidelines_df[guidelines_df["question_id"] == question_id].iloc[0]

        statement = q["statement"]
        turns = ast.literal_eval(q["turns"])
        values = ast.literal_eval(q["values"])
        choices = ast.literal_eval(g["choices"])
        rubric = choices[0]["turns"][0]

        judge_prompt = env.render_template(
            "judge_open_question.jinja",
            statement=statement,
            turns=turns,
            values=values,
            answer=answer,
            rubric=rubric,
        )

        response = ollama.chat(
            model=JUDGE_MODEL,
            options={"temperature": 0},
            messages=[{"role": "user", "content": judge_prompt}],
        )
        content = response["message"]["content"]

        try:
            result = json.loads(_extract_json(content))
            scores = [_clean_score(s) for s in result.get("scores", [])]
            score_total = sum(scores)
        except Exception:
            scores = None
            score_total = None

        results.append({
            "question_id": question_id,
            "model": model,
            "scores": scores,
            "total_score": score_total,
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "eval_open_questions.csv"), index=False)
    print()
    return df


# ── Avaliação: Questões de Múltipla Escolha ───────────────────────────────────

def evaluate_multiple_choice() -> pd.DataFrame:
    """Avalia questões de múltipla escolha comparando a resposta do modelo com o gabarito."""
    with open(os.path.join(RESULTS_DIR, "multiple_choice.json"), encoding="utf-8") as f:
        answers = json.load(f)

    results = []

    for entry in answers:
        model_answer = entry["answer"].strip().upper()
        match = re.search(r"\b([A-D])\b", model_answer)
        extracted = match.group(1) if match else model_answer[:1]

        correct = entry["correct"].strip().upper()

        results.append({
            "question_id": entry["question_id"],
            "model": entry["model"],
            "answer": extracted,
            "correct": correct,
            "is_correct": extracted == correct,
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "eval_multiple_choice.csv"), index=False)
    return df


# ── Avaliação Comparativa (questões abertas) ──────────────────────────────────

def evaluate_comparative() -> pd.DataFrame:
    """Compara respostas de diferentes modelos nas questões abertas usando critérios qualitativos."""
    with open(os.path.join(RESULTS_DIR, "open_questions.json"), encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    questions = df["question_id"].unique()

    total = len(questions)
    results = []

    for i, q_id in enumerate(questions, 1):
        subset = df[df["question_id"] == q_id]

        _progress(i, total, f"q:{q_id}")

        answers = [(row["model"], row["answer"]) for _, row in subset.iterrows()]

        prompt = env.render_template(
            "judge_comparative.jinja",
            question=subset.iloc[0]["question"],
            answers=answers,
        )

        response = ollama.chat(
            model=JUDGE_MODEL,
            options={"temperature": 0},
            messages=[{"role": "user", "content": prompt}],
        )
        content = response["message"]["content"]

        try:
            scores = json.loads(_extract_json(content))
            for model, r in scores.items():
                final = 0.4 * r["argumentacao"] + 0.4 * r["precisao"] + 0.2 * r["coesao"]
                results.append({
                    "question_id": q_id,
                    "model": model,
                    "argumentacao": r["argumentacao"],
                    "precisao": r["precisao"],
                    "coesao": r["coesao"],
                    "final_score": final,
                })
        except Exception:
            pass

    df_result = pd.DataFrame(results)
    df_result.to_csv(os.path.join(RESULTS_DIR, "eval_comparative.csv"), index=False)
    print()
    return df_result


# ── Leaderboard ────────────────────────────────────────────────────────────────

def generate_leaderboard(
    df_open: pd.DataFrame,
    df_mc: pd.DataFrame,
    df_comparative: pd.DataFrame,
) -> None:
    """Gera o leaderboard consolidado e gráficos separados por tipo de avaliação."""
    open_avg = df_open.groupby("model")["total_score"].mean().rename("open_score")
    mc_accuracy = (df_mc.groupby("model")["is_correct"].mean() * 100).rename("mc_accuracy_%")
    comp_metrics = df_comparative.groupby("model")[
        ["argumentacao", "precisao", "coesao", "final_score"]
    ].mean()

    leaderboard = pd.concat([open_avg, mc_accuracy, comp_metrics], axis=1).fillna(0)
    leaderboard.to_csv(os.path.join(RESULTS_DIR, "leaderboard.csv"))

    print("\n=== LEADERBOARD ===")
    print(leaderboard)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # ── Linha 1: Questões Abertas ──────────────────────────────────────────
    ax_rubrica = fig.add_subplot(gs[0, 0])
    open_avg.plot(kind="bar", ax=ax_rubrica, color="#4C72B0")
    ax_rubrica.set_title("Rubrica")
    ax_rubrica.set_ylabel("Pontuação média")
    ax_rubrica.set_xlabel("")
    ax_rubrica.tick_params(axis="x", rotation=0)

    ax_comp = fig.add_subplot(gs[0, 1])
    comp_metrics[["argumentacao", "precisao", "coesao"]].plot(kind="bar", ax=ax_comp)
    ax_comp.set_title("Avaliação Comparativa (0-5)")
    ax_comp.set_ylabel("Nota média")
    ax_comp.set_ylim(0, 5)
    ax_comp.set_xlabel("")
    ax_comp.tick_params(axis="x", rotation=0)
    ax_comp.legend(["Argumentação", "Precisão", "Coesão"], fontsize=8)

    fig.text(0.5, 0.95, "Questões Abertas", ha="center", fontsize=13, fontweight="bold")

    # ── Linha 2: Múltipla Escolha (centralizado) ──────────────────────────
    ax_mc = fig.add_subplot(gs[1, :])
    mc_accuracy.plot(kind="bar", ax=ax_mc, color="#55A868", width=0.4)
    ax_mc.set_title("Múltipla Escolha — Acurácia (%)")
    ax_mc.set_ylabel("Acurácia (%)")
    ax_mc.set_ylim(0, 100)
    ax_mc.set_xlabel("")
    ax_mc.tick_params(axis="x", rotation=0)

    plt.suptitle("Comparação de Modelos - Avaliação OAB", fontsize=15, y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.show()


# ── Execução direta ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_open = evaluate_open_questions()
    df_mc = evaluate_multiple_choice()
    df_comparative = evaluate_comparative()
    generate_leaderboard(df_open, df_mc, df_comparative)
