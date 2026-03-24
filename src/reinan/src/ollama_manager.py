import ollama

class OllamaManager:
    """
    Gerenciador para integração e execução de LLMs locais via Ollama.
    """
    AVAILABLE_MODELS = [
        "llama3.2:3b",
        "gemma2:2b",
        "qwen2.5:3b"
    ]

    def __init__(self):
        self.client = ollama.Client()

    def generate_response(self, model: str, system_prompt: str, user_prompt: str) -> str:
        """
        Gera uma resposta síncrona do Ollama usando um modelo específico.
        """
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Modelo {model} não suportado. Modelos disponíveis: {', '.join(self.AVAILABLE_MODELS)}")

        response = self.client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response['message']['content']
