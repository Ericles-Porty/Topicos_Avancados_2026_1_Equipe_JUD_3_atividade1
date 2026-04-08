import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import load_dataset as ld
import run_models as rm
import evaluation as ev


def main():
    print("=" * 60)
    print("  ETAPA 1/3 — Carregando e preparando datasets")
    print("=" * 60)
    ld.prepare_my_questions()

    print("\n" + "=" * 60)
    print("  ETAPA 2/3 — Inferência com LLMs + Curadoria")
    print("=" * 60)
    rm.run_open_questions()
    rm.run_multiple_choice_questions()
    rm.run_curator_tasks()

    print("\n" + "=" * 60)
    print("  ETAPA 3/3 — Avaliação e Leaderboard")
    print("=" * 60)
    df_open = ev.evaluate_open_questions()
    df_mc = ev.evaluate_multiple_choice()
    df_comparative = ev.evaluate_comparative()
    df_cross = ev.evaluate_cross_metrics()
    ev.generate_leaderboard(df_open, df_mc, df_comparative, df_cross)

    print("\n" + "=" * 60)
    print("  Pipeline concluído com sucesso!")
    print("=" * 60)


if __name__ == "__main__":
    main()
