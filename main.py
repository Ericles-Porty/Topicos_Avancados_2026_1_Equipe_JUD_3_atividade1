import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import load_dataset as ld
import run_models as rm
import evaluation as ev


PIPELINE_STAGES = ["prepare", "mc", "open", "curator", "eval"]
STAGES = PIPELINE_STAGES + ["all", "rag-populate"]


def _header(label: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)


def run_stage(stage: str, use_rag: bool = False, top_k: int = 10) -> None:
    if stage == "rag-populate":
        _header("ETAPA — Indexação da legislação no ChromaDB (RAG)")
        rm.run_rag_populate()
        return

    rm.set_rag(use_rag)
    rm.set_top_k(top_k)
    if use_rag:
        _header(f"RAG HABILITADO (top_k={top_k}) — respostas salvas com sufixo _rag")

    selected = set(PIPELINE_STAGES if stage == "all" else [stage])

    if "prepare" in selected:
        _header("ETAPA — Carregando e preparando datasets")
        ld.prepare_my_questions()

    if "mc" in selected:
        _header("ETAPA — Inferência: Múltipla Escolha")
        rm.run_multiple_choice_questions()

    if "open" in selected:
        _header("ETAPA — Inferência: Questões Abertas (rubrica)")
        rm.run_open_questions()

    if "curator" in selected:
        _header("ETAPA — Curadoria (dificuldade + legislação + subdomínio)")
        rm.run_curator_tasks()

    if "eval" in selected:
        _header("ETAPA — Avaliação e Leaderboard")
        df_open = ev.evaluate_open_questions()
        df_mc = ev.evaluate_multiple_choice()
        df_comparative = ev.evaluate_comparative()
        df_cross = ev.evaluate_cross_metrics()
        ev.generate_leaderboard(df_open, df_mc, df_comparative, df_cross)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline da Atividade 1. Use --stage para executar apenas uma etapa. "
            "Use --rag para inferência com Retrieval-Augmented Generation; antes, "
            "rode `--stage rag-populate` uma vez para indexar a legislação."
        )
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="all",
        help="Estágio a executar (default: all). Use 'rag-populate' para indexar a base RAG.",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Habilita o RAG na inferência (mc/open). Salva os resultados em *_rag.json.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Quantidade de trechos de lei recuperados pelo RAG (default: 10).",
    )
    args = parser.parse_args()

    run_stage(args.stage, use_rag=args.rag, top_k=args.top_k)

    _header("Pipeline concluído com sucesso!")


if __name__ == "__main__":
    main()
