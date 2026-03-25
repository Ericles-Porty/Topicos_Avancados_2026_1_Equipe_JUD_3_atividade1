import evaluate
from typing import Dict, Any
from datasets import load_dataset

class EvaluationManager:
    """
    Classe responsável por executar a avaliação das inferências
    utilizando a biblioteca Hugging Face evaluate.
    """
    def __init__(self, dataset_manager, storage_manager):
        self.dataset_manager = dataset_manager
        self.storage_manager = storage_manager
        
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")
        
    def _build_reference_map(self) -> Dict[str, str]:
        """
        Constrói um dicionário mapeando question_id -> texto de referência 
        baseado no conjunto guidelines do dataset oab_bench.
        """

        ds_guidelines = load_dataset("maritaca-ai/oab-bench", "guidelines")
        ref_map = {}
        for item in ds_guidelines['train']:
            q_id = item['question_id']
            choices = item.get('choices', [])
            ref_map[q_id] = "\n".join(str(c) for c in choices) if isinstance(choices, list) else str(choices)
            
        return ref_map

    def evaluate_results(self, dataset_name: str, model_name: str) -> Dict[str, Any]:
        """
        Lê os resultados inferidos do cache e calcula as métricas.
        """
        filename = f"{dataset_name}_{model_name.replace(':', '-')}_results"
        results = self.storage_manager.load_data(filename, fmt="json", sub_dir="results")
        
        if dataset_name == "oab_bench":
            ref_map = self._build_reference_map()
        else:
            raise ValueError(f"Avaliação não implementada para o dataset: {dataset_name}")
            
        predictions = []
        references = []
        
        for res in results:
            q_id = res['question_id']
            pred = res.get('ollama_response', "")
            ref = ref_map.get(q_id, "")
            
            predictions.append(pred)
            references.append(ref)
            
        # Calcula BLEU
        bleu_score = self.bleu.compute(predictions=predictions, references=references)
        
        # Calcula ROUGE
        rouge_score = self.rouge.compute(predictions=predictions, references=references)
        
        # Calcula BERTScore
        bert_score_raw = self.bertscore.compute(predictions=predictions, references=references, lang="pt")
        
        # Resumir BERTScore (calcular apenas a média do F1)
        bert_f1_mean = sum(bert_score_raw['f1']) / len(bert_score_raw['f1']) if bert_score_raw['f1'] else 0.0
        
        final_scores = {
            "bleu": bleu_score.get("bleu", 0.0),
            "rouge1": rouge_score.get("rouge1", 0.0),
            "rouge2": rouge_score.get("rouge2", 0.0),
            "rougeL": rouge_score.get("rougeL", 0.0),
            "bertscore_f1": bert_f1_mean
        }
        
        return final_scores
