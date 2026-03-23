from datasets import load_dataset

def main():
    load_dataset_oab_bench()
    load_dataset_oab_exams()

def load_dataset_oab_bench():
    ds_guidelines = load_dataset("maritaca-ai/oab-bench", "guidelines")
    ds_questions = load_dataset("maritaca-ai/oab-bench", "questions")
    
    questions = list(ds_guidelines['train']) + list(ds_questions['train'])
    questions_reinan = questions[176:188]
    
    print(f"Foram selecionadas {len(questions_reinan)} questões para o lote.")
    print("Conjuntos de dados carregados com sucesso!")

def load_dataset_oab_exams():
    ds_exams = load_dataset("eduagarcia/oab_exams")
    
    questions = list(ds_exams['train'])
    questions_reinan = questions[1845:1967]

    print(f"Foram selecionadas {len(questions_reinan)} questões para o lote.")
    print("Conjuntos de dados carregados com sucesso!")

if __name__ == "__main__":
    main()
