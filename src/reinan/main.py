import typer
from data.dataset_manager import DatasetManager

app = typer.Typer(help="CLI para manipulação de datasets da OAB.", no_args_is_help=True)
dataset_manager = DatasetManager()

@app.command()
def load_oab_bench():
    """
    Carrega e seleciona um lote de questões do dataset maritaca-ai/oab-bench.
    """
    questions_reinan = dataset_manager.load_oab_bench()
    
    typer.echo(f"Foram selecionadas {len(questions_reinan)} questões para o lote.")
    typer.echo("Conjuntos de dados carregados com sucesso!")

@app.command()
def load_oab_exams():
    """
    Carrega e seleciona um lote de questões do dataset eduagarcia/oab_exams.
    """
    questions_reinan = dataset_manager.load_oab_exams()

    typer.echo(f"Foram selecionadas {len(questions_reinan)} questões para o lote.")
    typer.echo("Conjuntos de dados carregados com sucesso!")

if __name__ == "__main__":
    app()
