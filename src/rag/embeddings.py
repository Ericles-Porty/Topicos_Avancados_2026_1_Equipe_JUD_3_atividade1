import httpx
import ollama
from typing import List


class OllamaEmbeddingProvider:
    """
    Garante a conectividade com o Ollama para a geração de embeddings locais.
    Auto-detecta e faz o download (pull) do modelo de embeddings caso ele não esteja presente.
    """

    def __init__(
        self,
        model_name: str = "qwen3-embedding:8b",
        host: str = "http://127.0.0.1:11434",
    ):
        self.model_name = model_name
        self.client = ollama.Client(host=host, timeout=httpx.Timeout(300.0))
        self._ensure_model_exists()

    def _ensure_model_exists(self) -> None:
        """Verifica se o modelo está presente localmente; caso contrário, faz o pull."""
        try:
            models_list = self.client.list()
            downloaded_models = [m.model for m in models_list.models]
            # O nome do modelo retornado por list() pode ter tags (ex: qwen3-embedding:8b:latest)
            if not any(self.model_name in m for m in downloaded_models):
                print(
                    f"[OllamaEmbeddingProvider] Modelo '{self.model_name}' não encontrado localmente."
                )
                print(
                    f"[OllamaEmbeddingProvider] Iniciando o download (pull) do modelo '{self.model_name}'..."
                )
                self.client.pull(self.model_name)
                print("[OllamaEmbeddingProvider] Download concluído com sucesso!")
        except Exception as e:
            # Caso ocorra algum erro (como falha de rede temporária), avisamos mas tentamos continuar
            print(f"[OllamaEmbeddingProvider] Aviso ao verificar/baixar modelo: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Gera embeddings para uma lista de documentos de forma robusta.
        Se a lista for muito grande, o Ollama pode falhar ou estourar memória;
        então processamos em blocos pequenos (batching).
        """
        if not texts:
            return []

        embeddings = []
        batch_size = 32  # Tamanho seguro para batches de texto longo no Ollama

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = self.client.embed(model=self.model_name, input=batch)
                embeddings.extend(response["embeddings"])
            except Exception as _e:
                # Tratamento de erro e retry para tentar isolar o problema
                print(
                    f"[OllamaEmbeddingProvider] Falha ao gerar batch {i // batch_size}. Retentando individualmente..."
                )
                for text in batch:
                    try:
                        resp = self.client.embed(model=self.model_name, input=text)
                        embeddings.append(resp["embeddings"][0])
                    except Exception as err:
                        print(
                            f"[OllamaEmbeddingProvider] Falha crítica ao gerar embedding individual: {err}"
                        )
                        raise RuntimeError(
                            f"Falha ao gerar embedding para o texto. Erro original: {err}"
                        ) from err

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Gera embedding para um único texto de consulta."""
        try:
            response = self.client.embed(model=self.model_name, input=text)
            return response["embeddings"][0]
        except Exception as e:
            raise RuntimeError(f"Erro ao gerar embedding da query: {e}")
