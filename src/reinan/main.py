import typer
from src.dataset_manager import DatasetManager
from src.storage_manager import StorageManager
from src.ollama_manager import OllamaManager
from src.execution_manager import ExecutionManager
from src.evaluation_manager import EvaluationManager

app = typer.Typer(no_args_is_help=True)
dataset_manager = DatasetManager()
storage_manager = StorageManager()
ollama_manager = OllamaManager()
execution_manager = ExecutionManager(dataset_manager, storage_manager, ollama_manager)
evaluation_manager = EvaluationManager(dataset_manager, storage_manager)

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

@app.command()
def run(
    dataset: str = typer.Argument(..., help="Nome do dataset para processar (ex: oab_bench, oab_exams)"),
    model: str = typer.Option(..., "--model", "-m", help=f"Modelo do ollama para execução: {', '.join(OllamaManager.AVAILABLE_MODELS)}"),
    limit: int = typer.Option(None, "--limit", "-l", help="Limitar a quantidade de questões a serem executadas.")
):
    """
    Executa a inferência nas questões do dataset através do LLM local conectado via Ollama.
    """

    if model not in OllamaManager.AVAILABLE_MODELS:
        typer.echo(f"Erro: Modelo '{model}' não é suportado pelo nosso OllamaManager.", err=True)
        raise typer.Exit(code=1)
        
    if dataset not in ["oab_bench", "oab_exams"]:
        typer.echo(f"Erro: Dataset '{dataset}' não reconhecido. Use 'oab_bench' ou 'oab_exams'.", err=True)
        raise typer.Exit(code=1)

    questions = execution_manager.get_questions(dataset, limit)

    typer.echo(f"Iniciando a execução de {len(questions)} questões no modelo {model}...")
    results = []
    
    with typer.progressbar(questions, label="Processando inferências") as progress:
        for q in progress:
            q_result = execution_manager.process_question(q, model, dataset)
            results.append(q_result)
            
    output_path = execution_manager.save_results(results, dataset, model)
    
    typer.echo("Inferência finalizada com sucesso!")
    typer.echo(f"Resultados salvos em: {output_path}")

@app.command()
def evaluate(
    dataset: str = typer.Argument(..., help="Nome do dataset para avaliar (ex: oab_bench)"),
    model: str = typer.Option(..., "--model", "-m", help=f"Modelo do ollama usado na execução: {', '.join(OllamaManager.AVAILABLE_MODELS)}")
):
    """
    Avalia as respostas geradas comparando-as com o gabarito oficial através do Hugging Face evaluate.
    """
    if model not in OllamaManager.AVAILABLE_MODELS:
        typer.echo(f"Erro: Modelo '{model}' não é suportado.", err=True)
        raise typer.Exit(code=1)
        
    if dataset not in ["oab_bench"]:
        typer.echo("Erro: Atualmente a avaliação está implementada apenas para 'oab_bench'.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Iniciando a avaliação do dataset {dataset} com modelo {model}...")
    
    try:
        scores = evaluation_manager.evaluate_results(dataset, model)
        typer.echo("\n--- Resultados da Avaliação ---")
        for metric, score in scores.items():
            typer.echo(f"{metric.upper()}: {score:.4f}")
    except Exception as e:
        typer.echo(f"Erro durante a avaliação: {e}", err=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
