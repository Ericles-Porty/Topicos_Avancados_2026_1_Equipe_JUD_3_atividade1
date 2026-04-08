import ast
import itertools
import json
import os
import re

import matplotlib.pyplot as plt
import ollama
import pandas as pd
import logging
from minijinja import Environment
from sklearn.metrics import precision_score, recall_score, f1_score

import load_dataset as ld

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
env = Environment(loader=lambda name: open(os.path.join(TEMPLATES_DIR, name), encoding="utf-8").read())

JUDGE_MODEL = "llama3.2:3b"
OLLAMA_TIMEOUT = 120.0
client = ollama.Client(timeout=OLLAMA_TIMEOUT)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _progress(current: int, total: int, label: str = "") -> None:
    pct = current / total * 100
    print(f"\r  [{current}/{total}] ({pct:.1f}%) {label}".ljust(80), end="", flush=True)



def _extract_json(text: str) -> str | None:
    """Extrai o primeiro objeto JSON encontrado no texto."""
    text = text.replace("```json", "").replace("```", "").strip()
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

        try:
            response = client.chat(
                model=JUDGE_MODEL,
                options={"temperature": 0},
                messages=[{"role": "user", "content": judge_prompt}],
            )
            content = response["message"]["content"]
        except Exception as e:
            logger.warning("Ollama falhou em q:%s model:%s — %s", question_id, model, e)
            results.append({
                "question_id": question_id, "model": model,
                "scores": None, "total_score": None,
            })
            continue

        max_score = sum(values)

        try:
            result = json.loads(_extract_json(content))
            raw_scores = [_clean_score(s) for s in result.get("scores", [])]
            scores = [min(s, v) for s, v in zip(raw_scores, values)]
            score_total = sum(scores)
            score_normalized = (score_total / max_score * 100) if max_score > 0 else 0.0
        except Exception as e:
            logger.warning("JSON inválido em q:%s model:%s — %s\nResposta: %s", question_id, model, e, content[:200])
            scores = None
            score_total = None
            score_normalized = None

        results.append({
            "question_id": question_id,
            "model": model,
            "scores": scores,
            "total_score": score_total,
            "max_score": max_score,
            "score_normalized": score_normalized,
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
        model_answer = entry["answer"].strip()
        extracted = None

        json_str = _extract_json(model_answer)
        if json_str:
            try:
                resp = json.loads(json_str)
                extracted = resp.get("resposta", "").strip().upper()[:1]
            except Exception:
                pass

        if not extracted or extracted not in "ABCD":
            match = re.search(r"\b([A-D])\b", model_answer.upper())
            extracted = match.group(1) if match else model_answer[:1].upper()

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

        try:
            response = client.chat(
                model=JUDGE_MODEL,
                options={"temperature": 0},
                messages=[{"role": "user", "content": prompt}],
            )
            content = response["message"]["content"]
        except Exception as e:
            logger.warning("Ollama falhou na comparativa q:%s — %s", q_id, e)
            continue

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
        except Exception as e:
            logger.warning("JSON inválido na comparativa q:%s — %s\nResposta: %s", q_id, e, content[:200])

    df_result = pd.DataFrame(results)
    df_result.to_csv(os.path.join(RESULTS_DIR, "eval_comparative.csv"), index=False)
    print()
    return df_result


# ── Métricas Cross-Model (BLEU, ROUGE, BERTScore) ────────────────────────────

def evaluate_cross_metrics() -> pd.DataFrame:
    """Compara respostas entre pares de modelos usando métricas automatizadas."""
    import evaluate as hf_evaluate

    with open(os.path.join(RESULTS_DIR, "open_questions.json"), encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    models = df["model"].unique().tolist()
    pairs = list(itertools.combinations(models, 2))

    bleu_metric = hf_evaluate.load("bleu")
    rouge_metric = hf_evaluate.load("rouge")
    bertscore_metric = hf_evaluate.load("bertscore")

    total = len(pairs)
    results = []

    for i, (model_a, model_b) in enumerate(pairs, 1):
        _progress(i, total, f"{model_a} vs {model_b}")

        df_a = df[df["model"] == model_a].set_index("question_id")
        df_b = df[df["model"] == model_b].set_index("question_id")
        common = df_a.index.intersection(df_b.index)

        refs = [df_a.loc[q, "answer"] for q in common]
        preds = [df_b.loc[q, "answer"] for q in common]

        bleu = bleu_metric.compute(predictions=preds, references=refs)
        rouge = rouge_metric.compute(predictions=preds, references=refs)
        bert = bertscore_metric.compute(predictions=preds, references=refs, lang="pt")

        results.append({
            "pair": f"{model_a} vs {model_b}",
            "bleu": bleu["bleu"],
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
            "bertscore_f1": sum(bert["f1"]) / len(bert["f1"]),
        })

    df_result = pd.DataFrame(results)
    df_result.to_csv(os.path.join(RESULTS_DIR, "eval_cross_metrics.csv"), index=False)
    print()
    return df_result


# ── Leaderboard ────────────────────────────────────────────────────────────────

def generate_leaderboard(
    df_open: pd.DataFrame,
    df_mc: pd.DataFrame,
    df_comparative: pd.DataFrame,
    df_cross: pd.DataFrame,
) -> None:
    """Gera o leaderboard consolidado e gráficos separados por tipo de avaliação."""

    # ── Métricas por modelo ───────────────────────────────────────────────
    open_avg = df_open.groupby("model")["score_normalized"].mean().rename("open_score_%")

    mc_accuracy = (df_mc.groupby("model")["is_correct"].mean() * 100).rename("mc_accuracy_%")

    # sklearn: precision, recall, f1 por modelo
    letter_to_int = {"A": 1, "B": 2, "C": 3, "D": 4}
    mc_sklearn = {}
    for model, grp in df_mc.groupby("model"):
        y_true = [letter_to_int.get(c, 0) for c in grp["correct"]]
        y_pred = [letter_to_int.get(a, 0) for a in grp["answer"]]
        mc_sklearn[model] = {
            "mc_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "mc_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "mc_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }
    df_sklearn = pd.DataFrame(mc_sklearn).T

    comp_metrics = df_comparative.groupby("model")[
        ["argumentacao", "precisao", "coesao", "final_score"]
    ].mean()

    leaderboard = pd.concat([open_avg, mc_accuracy, df_sklearn, comp_metrics], axis=1).fillna(0)
    leaderboard.to_csv(os.path.join(RESULTS_DIR, "leaderboard.csv"))

    print("\n=== LEADERBOARD ===")
    print(leaderboard)

    # ── Cross-metrics (pares de modelos — tabela separada) ────────────────
    print("\n=== MÉTRICAS CROSS-MODEL ===")
    print(df_cross.to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3)

    # Linha 1: Questões Abertas
    ax_rubrica = fig.add_subplot(gs[0, 0])
    open_avg.plot(kind="bar", ax=ax_rubrica, color="#4C72B0")
    ax_rubrica.set_title("Rubrica (normalizada)")
    ax_rubrica.set_ylabel("Aproveitamento médio (%)")
    ax_rubrica.set_ylim(0, 100)
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

    fig.text(0.5, 0.97, "Questões Abertas", ha="center", fontsize=13, fontweight="bold")

    # Linha 2: Cross-Metrics (BLEU, ROUGE, BERTScore)
    ax_cross = fig.add_subplot(gs[1, :])
    cross_plot = df_cross.set_index("pair")[["bleu", "rouge1", "rougeL", "bertscore_f1"]]
    cross_plot.plot(kind="bar", ax=ax_cross)
    ax_cross.set_title("Métricas Cross-Model (Questões Abertas)")
    ax_cross.set_ylabel("Score")
    ax_cross.set_ylim(0, 1)
    ax_cross.set_xlabel("")
    ax_cross.tick_params(axis="x", rotation=0)
    ax_cross.legend(["BLEU", "ROUGE-1", "ROUGE-L", "BERTScore F1"], fontsize=8)

    # Linha 3: Múltipla Escolha (accuracy + precision + recall + f1)
    ax_mc = fig.add_subplot(gs[2, :])
    mc_all = pd.concat([mc_accuracy / 100, df_sklearn], axis=1)
    mc_all.columns = ["Acurácia", "Precision", "Recall", "F1"]
    mc_all.plot(kind="bar", ax=ax_mc)
    ax_mc.set_title("Múltipla Escolha — Métricas de Classificação")
    ax_mc.set_ylabel("Score")
    ax_mc.set_ylim(0, 1)
    ax_mc.set_xlabel("")
    ax_mc.tick_params(axis="x", rotation=0)
    ax_mc.legend(fontsize=8)

    plt.suptitle("Comparação de Modelos - Avaliação OAB", fontsize=15, y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.show()


# ── Execução direta ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_open = evaluate_open_questions()
    df_mc = evaluate_multiple_choice()
    df_comparative = evaluate_comparative()
    df_cross = evaluate_cross_metrics()
    generate_leaderboard(df_open, df_mc, df_comparative, df_cross)
