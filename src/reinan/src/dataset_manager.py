from datasets import load_dataset
from typing import List, Dict, Any

class DatasetManager:
    """
    Classe responsável por gerenciar a carga e o pré-processamento
    dos datasets da OAB utilizados na aplicação.
    """
    def __init__(self):
        pass

    def load_oab_bench(self) -> List[Dict[str, Any]]:
        """
        Baixa as coleções de guidelines e questions do maritaca-ai/oab-bench,
        agrupa-as e retorna o lote de questões designadas.
        """
        ds_guidelines = load_dataset("maritaca-ai/oab-bench", "guidelines")
        ds_questions = load_dataset("maritaca-ai/oab-bench", "questions")
        
        questions = list(ds_guidelines['train']) + list(ds_questions['train'])
        return questions[176:188]

    def load_oab_exams(self) -> List[Dict[str, Any]]:
        """
        Baixa as questões do eduagarcia/oab_exams e retorna o lote designado.
        """
        ds_exams = load_dataset("eduagarcia/oab_exams")
        
        questions = list(ds_exams['train'])
        return questions[1845:1967]
