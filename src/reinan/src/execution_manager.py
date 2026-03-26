from typing import List, Dict, Any
from pathlib import Path
import jinja2
import json

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

    def _render_prompt(self, dataset: str, template_name: str, context: dict) -> str:
        """
        Função auxiliar para ler o arquivo .minijinja do disco e renderizá-lo com Jinja2.
        """
        
        prompts_dir = Path(__file__).parent.parent / "prompts"
        template_path = prompts_dir / dataset / template_name
        
        if not template_path.exists():
            return ""
            
        with open(template_path, "r", encoding="utf-8") as f:
            template_str = f.read()
            
        template = jinja2.Template(template_str)
        return template.render(**context)

    def process_question(self, q: Dict[str, Any], model: str, dataset: str) -> Dict[str, Any]:
        """
        Recebe uma questão crua, carrega seus templates locais minijinja, formata o prompt usando Jinja2 
        e extrai a resposta do LLM.
        """

        user_prompt = self._render_prompt(dataset, "user_template.minijinja", q)
        system_prompt = self._render_prompt(dataset, "system_template.minijinja", q)
        
        if not user_prompt:
            user_prompt = q.get("statement", q.get("question", ""))
            if "choices" in q:
                choices_text = [f"{label}) {text}" for label, text in zip(q['choices']['label'], q['choices']['text'])]
                user_prompt += "\n" + "\n".join(choices_text)
                
        if not system_prompt:
            system_prompt = q.get("system", "Você é um assistente prestativo especialista em direito brasileiro.")
        
        try:
            response = self.ollama_manager.generate_response(model, system_prompt, user_prompt)
        except Exception as e:
            response = f"ERRO: {e}"
            
        q_result = q.copy()
        q_result['ollama_response'] = response
        q_result['model_used'] = model

        if dataset == "oab_exams":
            resp_str_clean = response.replace("```json", "").replace("```", "").strip()
            resp_json = json.loads(resp_str_clean)

            q_result['objective_answer'] = resp_json.get("resposta_objetiva", "")
        
        return q_result

    def save_results(self, results: List[Dict[str, Any]], dataset: str, model: str) -> Path:
        """
        Salva os resultados consolidados na subpasta definida para o cache.
        """
        filename = f"{dataset}_{model.replace(':', '-')}_results"
        output_path = self.storage_manager.save_data(results, filename, fmt="json", sub_dir="results")
        return output_path
