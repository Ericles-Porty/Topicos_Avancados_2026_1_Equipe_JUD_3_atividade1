import typer
from data.dataset_manager import DatasetManager
from data.storage_manager import StorageManager

app = typer.Typer(no_args_is_help=True)
dataset_manager = DatasetManager()
storage_manager = StorageManager()

@app.callback()
def main_callback():
    """
    CLI para manipulação de datasets da OAB.
    """
    pass

@app.command()
def pull(
    dataset: str = typer.Argument(..., help="Nome do dataset para baixar (ex: oab_bench, oab_exams)"), 
    output: str = typer.Option("json", "--output", help="Formato do arquivo de saída (json ou csv)")
):
    """
    Baixa as informações do dataset e salva no diretório de armazenamento local em JSON ou CSV.
    """
    if dataset == "oab_bench":
        questions = dataset_manager.load_oab_bench()
    elif dataset == "oab_exams":
        questions = dataset_manager.load_oab_exams()
    else:
        typer.echo(f"Erro: Dataset '{dataset}' não reconhecido. Use 'oab_bench' ou 'oab_exams'.", err=True)
        raise typer.Exit(code=1)
        
    caminho_arquivo = storage_manager.save_data(questions, dataset, fmt=output, sub_dir="dataset")
    
    typer.echo(f"Foram selecionadas {len(questions)} questões para o lote.")
    typer.echo(f"Conjunto de dados salvo com sucesso em: {caminho_arquivo}")

if __name__ == "__main__":
    app()
