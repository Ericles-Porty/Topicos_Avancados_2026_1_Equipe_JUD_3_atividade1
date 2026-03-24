from typing import List, Dict, Any
from pathlib import Path

class ExecutionManager:
    """
    Gerenciador responsável por orquestrar o processo de inferência.
    """
    def __init__(self, dataset_manager, storage_manager, ollama_manager):
        self.dataset_manager = dataset_manager
        self.storage_manager = storage_manager
        self.ollama_manager = ollama_manager

    def get_questions(self, dataset: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Carrega as questões do dataset especificado e aplica o limite caso exista.
        """
        if dataset == "oab_bench":
            questions = self.dataset_manager.load_oab_bench()
        elif dataset == "oab_exams":
            questions = self.dataset_manager.load_oab_exams()
        else:
            raise ValueError(f"Dataset desconhecido: {dataset}")

        if limit is not None and limit > 0:
            questions = questions[:limit]
            
        return questions

    def process_question(self, q: Dict[str, Any], model: str) -> Dict[str, Any]:
        """
        Recebe uma questão crua, formata o prompt e extrai a resposta do LLM.
        """
        system_prompt = q.get("system", "Você é um assistente prestativo especialista em direito brasileiro.")
        user_prompt = q.get("statement", q.get("question", ""))
        
        if "choices" in q:
            choices_text = [f"{label}) {text}" for label, text in zip(q['choices']['label'], q['choices']['text'])]
            user_prompt += "\n" + "\n".join(choices_text)
        
        try:
            response = self.ollama_manager.generate_response(model, system_prompt, user_prompt)
        except Exception as e:
            response = f"ERRO: {e}"
            
        q_result = q.copy()
        q_result['ollama_response'] = response
        q_result['model_used'] = model
        
        return q_result

    def save_results(self, results: List[Dict[str, Any]], dataset: str, model: str) -> Path:
        """
        Salva os resultados consolidados na subpasta definida para o cache.
        """
        filename = f"{dataset}_{model.replace(':', '-')}_results"
        output_path = self.storage_manager.save_data(results, filename, fmt="json", sub_dir="results")
        return output_path
