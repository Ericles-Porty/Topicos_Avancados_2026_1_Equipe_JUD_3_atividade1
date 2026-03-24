import typer
from data.dataset_manager import DatasetManager
from data.storage_manager import StorageManager
from data.ollama_manager import OllamaManager

app = typer.Typer(no_args_is_help=True)
dataset_manager = DatasetManager()
storage_manager = StorageManager()
ollama_manager = OllamaManager()

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
    dataset: str = typer.Argument(..., help="Nome do dataset para ler do cache e processar (ex: oab_bench)"),
    model: str = typer.Option(..., "--model", "-m", help=f"Modelo do ollama para execução: {', '.join(OllamaManager.AVAILABLE_MODELS)}")
):
    """
    Executa a inferência nas questões do dataset através do LLM local conectado via Ollama.
    """
    if model not in OllamaManager.AVAILABLE_MODELS:
        typer.echo(f"Erro: Modelo '{model}' não é suportado pelo nosso OllamaManager.", err=True)
        raise typer.Exit(code=1)

    questions = dataset_manager.load_oab_bench()

    typer.echo(f"Iniciando a execução de {len(questions)} questões no modelo {model}...")
    results = []
    
    with typer.progressbar(questions, label="Processando inferências") as progress:
        for q in progress:
            system_prompt = q.get("system", "Você é um assistente prestativo especialista em direito brasileiro.")
            user_prompt = q.get("statement", q.get("question", ""))
            
            if "choices" in q:
                choices_text = [f"{label}) {text}" for label, text in zip(q['choices']['label'], q['choices']['text'])]
                user_prompt += "\n" + "\n".join(choices_text)
            
            try:
                response = ollama_manager.generate_response(model, system_prompt, user_prompt)
            except Exception as e:
                response = f"ERRO: {e}"
                
            q_result = q.copy()
            q_result['ollama_response'] = response
            q_result['model_used'] = model
            results.append(q_result)
            
    filename = f"{dataset}_{model.replace(':', '-')}_results"
    output_path = storage_manager.save_data(results, filename, fmt="json", sub_dir="results")
    
    typer.echo("Inferência finalizada com sucesso!")
    typer.echo(f"Resultados salvos em: {output_path}")

if __name__ == "__main__":
    app()
