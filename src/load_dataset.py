import os
import requests
import pandas as pd
from datasets import load_dataset as hf_load_dataset

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")

OPEN_DIR            = os.path.join(DATASET_DIR, "open_questions")
MULTIPLE_CHOICE_DIR = os.path.join(DATASET_DIR, "multiple_choice")

MY_QUESTIONS_DIR    = os.path.join(DATASET_DIR, "my_questions")

os.makedirs(OPEN_DIR, exist_ok=True)
os.makedirs(MULTIPLE_CHOICE_DIR, exist_ok=True)
os.makedirs(MY_QUESTIONS_DIR, exist_ok=True)


# ── Questões Abertas ───────────────────────────────────────────────────────────

_OPEN_QUESTIONS_URL  = "https://raw.githubusercontent.com/maritaca-ai/oab-bench/refs/heads/main/data/oab_bench/question.jsonl"
_OPEN_GUIDELINES_URL = "https://raw.githubusercontent.com/maritaca-ai/oab-bench/refs/heads/main/data/oab_bench/reference_answer/guidelines.jsonl"

_OPEN_QUESTIONS_PATH  = os.path.join(OPEN_DIR, "questions.jsonl")
_OPEN_GUIDELINES_PATH = os.path.join(OPEN_DIR, "guidelines.jsonl")


def _progress(current: int, total: int, label: str = "") -> None:
    pct = current / total * 100
    print(f"\r  [{current}/{total}] ({pct:.1f}%) {label}".ljust(80), end="", flush=True)


def _download_file(url: str, dest: str) -> None:
    response = requests.get(url)
    response.raise_for_status()
    with open(dest, "wb") as f:
        f.write(response.content)


def load_open_questions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega questões abertas e gabaritos do OAB-Bench, baixando os arquivos se necessário."""
    if not os.path.exists(_OPEN_QUESTIONS_PATH):
        _download_file(_OPEN_QUESTIONS_URL, _OPEN_QUESTIONS_PATH)
    if not os.path.exists(_OPEN_GUIDELINES_PATH):
        _download_file(_OPEN_GUIDELINES_URL, _OPEN_GUIDELINES_PATH)

    df_questions = pd.read_json(_OPEN_QUESTIONS_PATH, lines=True)
    df_guidelines = pd.read_json(_OPEN_GUIDELINES_PATH, lines=True)

    df_questions.to_csv(os.path.join(OPEN_DIR, "questions.csv"), index=False)
    df_guidelines.to_csv(os.path.join(OPEN_DIR, "guidelines.csv"), index=False)

    return df_questions, df_guidelines


# ── Questões de Múltipla Escolha ───────────────────────────────────────────────

_MULTIPLE_CHOICE_DATASET_ID = "eduagarcia/oab_exams"


def _escape_newlines(df: pd.DataFrame) -> pd.DataFrame:
    """Substitui quebras de linha reais por \\n literal em colunas de texto."""
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.replace("\n", "\\n", regex=False)
    return df


def _flatten_choices(df: pd.DataFrame) -> pd.DataFrame:
    """Expande a coluna 'choices' (dict com 'label' e 'text') em colunas separadas."""
    expanded = df["choices"].apply(
        lambda c: {f"choice_{l.lower()}": t for l, t in zip(c["label"], c["text"])}
    )
    return pd.concat([df.drop(columns=["choices"]), pd.DataFrame(expanded.tolist(), index=df.index)], axis=1)


def load_multiple_choice_questions() -> dict[str, pd.DataFrame]:
    """Carrega questões de múltipla escolha do HuggingFace e salva cada split como CSV."""
    dataset = hf_load_dataset(_MULTIPLE_CHOICE_DATASET_ID)

    splits: dict[str, pd.DataFrame] = {}
    for split, data in dataset.items():
        df = _escape_newlines(_flatten_choices(data.to_pandas()))
        dest = os.path.join(MULTIPLE_CHOICE_DIR, f"{split}.csv")
        df.to_csv(dest, index=False)
        splits[split] = df

    return splits


# ── Preparação das questões selecionadas ───────────────────────────────────────

OPEN_SLICE = slice(153, 165)        # linhas 153–164 inclusive
MC_SLICE   = slice(1600, 1723)      # linhas 1600–1722 inclusive


def prepare_my_questions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os datasets completos e extrai o subconjunto de questões selecionadas."""
    total_steps = 4

    _progress(1, total_steps, "Carregando questões abertas...")
    df_questions, _ = load_open_questions()

    _progress(2, total_steps, "Carregando múltipla escolha (HuggingFace)...")
    splits_mc = load_multiple_choice_questions()

    _progress(3, total_steps, "Preparando subconjunto de questões abertas...")
    my_open = df_questions.iloc[OPEN_SLICE]
    my_open.to_csv(os.path.join(MY_QUESTIONS_DIR, "open_questions.csv"), index=False)

    _progress(4, total_steps, "Preparando subconjunto de múltipla escolha...")
    my_mc = splits_mc["train"].iloc[MC_SLICE]
    my_mc.to_csv(os.path.join(MY_QUESTIONS_DIR, "multiple_choice.csv"), index=False)

    print(f"\n\nQuestões abertas selecionadas:             {len(my_open)}")
    print(f"Questões de múltipla escolha selecionadas: {len(my_mc)}")

    return my_open, my_mc


# ── Execução direta ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    prepare_my_questions()
