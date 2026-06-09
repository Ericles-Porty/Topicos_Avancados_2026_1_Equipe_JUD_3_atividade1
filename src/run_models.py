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


# ── RAG (Retrieval-Augmented Generation) ───────────────────────────────────────
# Port fiel da implementação do Reinan (src/rag): chunking + embeddings +
# ChromaDB + busca híbrida (vetorial + lexical) + rerank + porta de confiança.

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
RAG_DIR = os.path.join(_PROJECT_ROOT, "database", "rag")
RAG_DB_PATH = os.path.join(_PROJECT_ROOT, ".chroma")
RAG_COLLECTION = "legislacao"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

_USE_RAG = False
_RAG_TOP_K = 10
_rag_builder = None


def set_rag(enabled: bool) -> None:
    """Habilita/desabilita o uso do RAG na inferência."""
    global _USE_RAG
    _USE_RAG = enabled


def set_top_k(top_k: int) -> None:
    """Define quantos trechos de lei o RAG injeta no prompt."""
    global _RAG_TOP_K
    _RAG_TOP_K = top_k


def _get_rag_builder():
    """Inicializa (uma vez) o banco vetorial e o construtor de contexto RAG."""
    global _rag_builder
    if _rag_builder is None:
        from rag.database import LegislationVectorDB
        from rag.embeddings import OllamaEmbeddingProvider
        from rag.context_builder import RagContextBuilder

        provider = OllamaEmbeddingProvider(model_name=EMBEDDING_MODEL)
        db = LegislationVectorDB(
            db_path=RAG_DB_PATH,
            collection_name=RAG_COLLECTION,
            embedding_provider=provider,
        )
        _rag_builder = RagContextBuilder(db, top_k=_RAG_TOP_K)
    return _rag_builder


def _augment_with_rag(user_content: str, q: dict, model: str):
    """Recupera a legislação de suporte e a prefixa ao prompt do usuário."""
    builder = _get_rag_builder()
    context_str, rag_info = builder.get_context_and_info(
        q, top_k=_RAG_TOP_K, model=model
    )
    if not context_str:
        return user_content, rag_info
    augmented = (
        "Considere a legislação de suporte abaixo obtida da base de conhecimento "
        "jurídica para responder à questão:\n"
        "[LEGISLAÇÃO DE SUPORTE]\n"
        f"{context_str}\n"
        "--- FIM DA LEGISLAÇÃO DE SUPORTE ---\n\n"
        f"{user_content}"
    )
    return augmented, rag_info


def run_rag_populate() -> None:
    """Indexa a legislação de database/rag/ no ChromaDB (chunking + embeddings)."""
    from pathlib import Path
    from rag.chunker import LegislationChunker
    from rag.embeddings import OllamaEmbeddingProvider
    from rag.database import LegislationVectorDB

    rag_dir = Path(RAG_DIR)
    html_files = sorted(rag_dir.glob("*.html"))
    if not html_files:
        print(f"Nenhum arquivo HTML encontrado em {rag_dir}.")
        return
    print(f"Encontrados {len(html_files)} arquivos para indexação.")

    chunker = LegislationChunker()
    all_chunks = []
    for fp in html_files:
        chunks = chunker.chunk_file(fp)
        all_chunks.extend(chunks)
        print(f"  {fp.name}: {len(chunks)} trechos")
    print(f"Total de chunks extraídos: {len(all_chunks)}")

    provider = OllamaEmbeddingProvider(model_name=EMBEDDING_MODEL)
    db = LegislationVectorDB(
        db_path=RAG_DB_PATH,
        collection_name=RAG_COLLECTION,
        embedding_provider=provider,
    )
    db.populate(all_chunks, reset=True)
    print("Indexação concluída no ChromaDB.")


def _progress(current: int, total: int, label: str = "") -> None:
    pct = current / total * 100
    print(f"\r  [{current}/{total}] ({pct:.1f}%) {label}".ljust(80), end="", flush=True)


# ── Questões Abertas ───────────────────────────────────────────────────────────

def run_open_questions() -> None:
    df = pd.read_csv(os.path.join(ld.MY_QUESTIONS_DIR, "open_questions.csv"))
    suffix = "_rag" if _USE_RAG else ""
    dest = os.path.join(RESULTS_DIR, f"open_questions{suffix}.json")

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

        for model in pending_models:
            step += 1
            _progress(step, total, f"q:{row['question_id']} | {model}")

            user_content = question
            rag_info = []
            if _USE_RAG:
                user_content, rag_info = _augment_with_rag(
                    question, {"statement": question}, model
                )

            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_content},
            ]
            for turn in turns:
                if turn.strip():
                    messages.append({"role": "user", "content": turn})

            try:
                response = client.chat(model=model, messages=messages)
                answer   = response["message"]["content"]
            except Exception as e:
                logger.warning("Ollama falhou em q:%s model:%s — %s", row["question_id"], model, e)
                answer = ""

            entry = {
                "question_id": row["question_id"],
                "model":       model,
                "question":    question,
                "answer":      answer,
            }
            if _USE_RAG:
                entry["used_rag"] = True
                entry["rag_info"] = rag_info
            results.append(entry)
            done.add((row["question_id"], model))

    with open(dest, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\nRespostas abertas salvas em {dest}")


# ── Questões de Múltipla Escolha ───────────────────────────────────────────────

CHOICE_LABELS = ["A", "B", "C", "D"]


def run_multiple_choice_questions() -> None:
    df = pd.read_csv(os.path.join(ld.MY_QUESTIONS_DIR, "multiple_choice.csv"))
    suffix = "_rag" if _USE_RAG else ""
    dest = os.path.join(RESULTS_DIR, f"multiple_choice{suffix}.json")

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
        base_user_prompt = env.render_template("multiple_choice.jinja", question=question, choices=choices)

        # Estrutura esperada pelo banco vetorial do Reinan: choices = {"label": [...], "text": [...]}
        rag_q = {
            "question": question,
            "choices": {
                "label": [label for label, _ in choices],
                "text": [text for _, text in choices],
            },
        }

        for model in pending_models:
            step += 1
            _progress(step, total, f"q:{row['id']} | {model}")

            user_prompt = base_user_prompt
            rag_info = []
            if _USE_RAG:
                user_prompt, rag_info = _augment_with_rag(base_user_prompt, rag_q, model)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            try:
                response = client.chat(model=model, messages=messages)
                answer   = response["message"]["content"]
            except Exception as e:
                logger.warning("Ollama falhou em q:%s model:%s — %s", row["id"], model, e)
                answer = ""

            entry = {
                "question_id": row["id"],
                "model":       model,
                "question":    question,
                "choices":     dict(choices),
                "answer":      answer,
                "correct":     row["answerKey"],
            }
            if _USE_RAG:
                entry["used_rag"] = True
                entry["rag_info"] = rag_info
            results.append(entry)
            done.add((row["id"], model))

    with open(dest, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\nRespostas de múltipla escolha salvas em {dest}")


# ── Curadoria (dificuldade + legislação) ──────────────────────────────────────

CURATOR_MODEL = "llama3.2:3b"

NIVEL_BY_DIFICULDADE = {
    1: "Nível 1 — Recuperação Factual Direta",
    2: "Nível 2 — Raciocínio Lógico-Dedutivo",
    3: "Nível 3 — Hermenêutica Jurídica Complexa",
}


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

        if not entry.get("nivel") and entry.get("dificuldade") in NIVEL_BY_DIFICULDADE:
            entry["nivel"] = NIVEL_BY_DIFICULDADE[entry["dificuldade"]]

        try:
            leg_json = json.loads(_extract_json(leg_resp["message"]["content"]))
            corpus = leg_json.get("corpus_referencia") or "Inconclusivo"
            entry["corpus_referencia"] = corpus
        except Exception as e:
            logger.warning("JSON inválido (corpus de referência) q:%s — %s", q["question_id"], e)
            entry["corpus_referencia"] = "Inconclusivo"

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
