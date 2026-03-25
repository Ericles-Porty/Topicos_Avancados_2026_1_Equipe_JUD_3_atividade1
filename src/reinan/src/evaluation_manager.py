import evaluate
from typing import Dict, Any

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

    def evaluate_cross_models(self, dataset_name: str, models: list) -> Dict[str, Dict[str, Any]]:
        """
        Executa a avaliação cruzada (pairwise) entre os modelos disponíveis.
        """
        if len(models) < 2:
            raise ValueError("São necessários pelo menos 2 modelos para rodar a avaliação cruzada.")
            
        all_results = {}
        for model in models:
            filename = f"{dataset_name}_{model.replace(':', '-')}_results"
            results = self.storage_manager.load_data(filename, fmt="json", sub_dir="results")
            model_responses = {res['question_id']: res.get('ollama_response', "") for res in results}
            all_results[model] = model_responses
            
        cross_scores = {}
        import itertools
        
        pairs = list(itertools.combinations(models, 2))
        for model_ref, model_pred in pairs:
            predictions = []
            references = []
            
            common_qs = set(all_results[model_ref].keys()).intersection(set(all_results[model_pred].keys()))
            
            for q_id in common_qs:
                ref_text = all_results[model_ref][q_id]
                pred_text = all_results[model_pred][q_id]
                
                references.append(ref_text)
                predictions.append(pred_text)
                
            if not predictions:
                continue
                
            bleu_score = self.bleu.compute(predictions=predictions, references=references)
            rouge_score = self.rouge.compute(predictions=predictions, references=references)
            bert_score_raw = self.bertscore.compute(predictions=predictions, references=references, lang="pt")
            
            bert_f1_mean = sum(bert_score_raw['f1']) / len(bert_score_raw['f1']) if bert_score_raw['f1'] else 0.0
            
            pair_name = f"{model_ref} vs {model_pred}"
            cross_scores[pair_name] = {
                "bleu": bleu_score.get("bleu", 0.0),
                "rouge1": rouge_score.get("rouge1", 0.0),
                "rouge2": rouge_score.get("rouge2", 0.0),
                "rougeL": rouge_score.get("rougeL", 0.0),
                "bertscore_f1": bert_f1_mean
            }
            
        return cross_scores
