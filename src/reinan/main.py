import typer
from datasets import load_dataset

app = typer.Typer(help="CLI para manipulação de datasets da OAB.", no_args_is_help=True)

@app.command()
def load_oab_bench():
    """
    Carrega e seleciona um lote de questões do dataset maritaca-ai/oab-bench.
    """
    ds_guidelines = load_dataset("maritaca-ai/oab-bench", "guidelines")
    ds_questions = load_dataset("maritaca-ai/oab-bench", "questions")
    
    questions = list(ds_guidelines['train']) + list(ds_questions['train'])
    questions_reinan = questions[176:188]
    
    typer.echo(f"Foram selecionadas {len(questions_reinan)} questões para o lote.")
    typer.echo("Conjuntos de dados carregados com sucesso!")

@app.command()
def load_oab_exams():
    """
    Carrega e seleciona um lote de questões do dataset eduagarcia/oab_exams.
    """
    ds_exams = load_dataset("eduagarcia/oab_exams")
    
    questions = list(ds_exams['train'])
    questions_reinan = questions[1845:1967]

    typer.echo(f"Foram selecionadas {len(questions_reinan)} questões para o lote.")
    typer.echo("Conjuntos de dados carregados com sucesso!")

if __name__ == "__main__":
    app()
