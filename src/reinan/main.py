def main():
    from datasets import load_dataset
    
    ds_guidelines = load_dataset("maritaca-ai/oab-bench", "guidelines")
    ds_questions = load_dataset("maritaca-ai/oab-bench", "questions")
    
    questions = list(ds_guidelines['train']) + list(ds_questions['train'])
    questions_reinan = questions[176:188]
    
    print(f"Foram selecionadas {len(questions_reinan)} questões para o lote.")
    print("Conjuntos de dados carregados com sucesso!")


if __name__ == "__main__":
    main()
