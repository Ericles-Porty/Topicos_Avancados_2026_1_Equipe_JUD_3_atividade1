import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import load_dataset as ld
import run_models as rm
import evaluation as ev


STAGES = ["prepare", "mc", "open", "curator", "eval", "all"]


def _header(label: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)


def run_stage(stage: str) -> None:
    selected = set(STAGES if stage == "all" else [stage])

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
            "Combine chamadas (ex.: `--stage prepare` e depois `--stage mc`) "
            "para rodar somente múltipla escolha sem reexecutar a rubrica."
        )
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="all",
        help="Estágio a executar (default: all).",
    )
    args = parser.parse_args()

    run_stage(args.stage)

    _header("Pipeline concluído com sucesso!")


if __name__ == "__main__":
    main()
