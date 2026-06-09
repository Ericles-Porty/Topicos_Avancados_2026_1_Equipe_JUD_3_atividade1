import re
import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from rag.embeddings import OllamaEmbeddingProvider
from rag.lexical import LexicalRetriever, expand_query, strip_accents


# ---------------------------------------------------------------------------
# Padrões heurísticos para detecção de trecho literal de lei
# ---------------------------------------------------------------------------
_LEGAL_LITERAL_PATTERNS = [
    r"^sao incapazes",
    r"^e nulo",
    r"^e anulavel",
    r"^compete a",
    r"^considera-se",
    r"^incumbe ao",
    r"^nao se admite",
    r"^e vedado",
    r"^e assegurado",
    r"^a lei",
    r"^o juiz",
    r"^o contrato",
    r"^a sentenca",
    r"^salvo disposicao",
    r"^ressalvado",
    r"^sao direitos",
    r"^constituem",
    r"^extingue-se",
    r"^prescreve em",
    r"^e defeso",
    r"^ninguem pode",
    r"^todo aquele",
    r"^aquele que",
]

_LEGAL_LITERAL_FRAGMENTS = [
    "nos termos",
    "na forma da lei",
    "na forma do",
    "salvo disposicao em contrario",
    "ressalvada a hipotese",
    "sob pena de",
    "sem prejuizo",
    "na forma prevista",
    "conforme disposto",
    "aplicam-se",
    "observado o disposto",
]

# ---------------------------------------------------------------------------
# Mapeamento de ramos jurídicos para detecção heurística
# ---------------------------------------------------------------------------
_BRANCH_KEYWORDS: Dict[str, List[str]] = {
    "Direito Civil": [
        "contrato",
        "obrigacao",
        "responsabilidade civil",
        "posse",
        "propriedade",
        "familia",
        "casamento",
        "divorcio",
        "heranca",
        "sucessao",
        "testamento",
        "capacidade",
        "incapacidade",
        "incapaz",
        "nulidade",
        "anulacao",
        "anulavel",
        "nulo",
        "prescricao",
        "decadencia",
        "dano moral",
        "indenizacao",
        "vicio",
        "eviccao",
        "usucapiao",
        "hipoteca",
        "obrigacoes",
        "codigo civil",
        "negocio juridico",
        "ato juridico",
        "coacao",
        "dolo",
        "erro",
        "estado de perigo",
        "lesao",
        "fraude",
        "simulacao",
        "pessoa natural",
        "personalidade",
        "doacao",
        "alimentos",
        "guarda",
        "adocao",
        "curatela",
        "tutela",
        "inventario",
        "partilha",
    ],
    "Direito Penal": [
        "crime",
        "delito",
        "pena",
        "prisao",
        "flagrante",
        "sancao penal",
        "reincidencia",
        "prescricao penal",
        "codigo penal",
        "dolo",
        "culpa",
        "tentativa",
        "consumacao",
        "agravante",
        "atenuante",
        "qualificadora",
        "tipo penal",
        "injuria",
        "difamacao",
        "calúnia",
        "furto",
        "roubo",
        "estelionato",
        "homicidio",
        "lesao corporal",
        "ameaca",
    ],
    "Direito do Trabalho": [
        "empregado",
        "empregador",
        "clt",
        "rescisao",
        "justa causa",
        "ferias",
        "salario",
        "fgts",
        "aviso previo",
        "estabilidade",
        "trabalho",
        "trabalhista",
        "sindicato",
        "convencao coletiva",
        "acordo coletivo",
        "horas extras",
        "intervalo",
        "adicional",
    ],
    "Direito Constitucional": [
        "constituicao",
        "direitos fundamentais",
        "mandado de seguranca",
        "habeas corpus",
        "acao popular",
        "controle de constitucionalidade",
        "competencia",
        "federalismo",
        "separacao de poderes",
        "stf",
        "stj",
        "direitos sociais",
        "emenda constitucional",
        "clausula petrea",
    ],
    "Direito Administrativo": [
        "licitacao",
        "pregao",
        "concorrencia",
        "administracao publica",
        "servidor publico",
        "improbidade",
        "ato administrativo",
        "poder de policia",
        "concessao",
        "permissao",
        "desapropriacao",
    ],
    "Direito Empresarial": [
        "sociedade",
        "empresa",
        "socio",
        "falencia",
        "recuperacao judicial",
        "titulo de credito",
        "duplicata",
        "marca",
        "patente",
        "propriedade industrial",
        "acao societaria",
    ],
    "Direito do Consumidor": [
        "consumidor",
        "fornecedor",
        "produto defeituoso",
        "relacao de consumo",
        "cdc",
        "codigo de defesa do consumidor",
        "vicio do produto",
    ],
}

# ---------------------------------------------------------------------------
# Termos jurídicos para extração de conceitos
# ---------------------------------------------------------------------------
_LEGAL_VERBS = [
    "anular",
    "cobrar",
    "prescrever",
    "impugnar",
    "denunciar",
    "adjudicar",
    "alienar",
    "revogar",
    "rescindir",
    "executar",
    "indenizar",
    "reparar",
    "restituir",
    "compensar",
    "opor",
    "recorrer",
    "embargar",
    "apelar",
    "contestar",
    "reconvir",
    "desapropriar",
    "usucapir",
]

_LEGAL_CONSEQUENCES = [
    "nulidade",
    "anulacao",
    "anulabilidade",
    "condenacao",
    "prescricao",
    "decadencia",
    "estabilidade",
    "legitimidade",
    "competencia",
    "indenizacao",
    "reparacao",
    "rescisao",
    "resolucao",
    "resilicao",
    "compensacao",
    "restituicao",
    "absolvicao",
    "extincao",
]

_LEGAL_FOUNDATIONS = [
    "erro",
    "dolo",
    "coacao",
    "estado de perigo",
    "lesao",
    "fraude",
    "simulacao",
    "incapacidade",
    "boa-fe",
    "ma-fe",
    "coisa julgada",
    "decadencia",
    "competencia",
    "legitimidade",
    "capacidade",
    "causa transitoria",
    "enfermidade",
    "deficiencia mental",
    "vicio de consentimento",
    "vicio redibitorio",
    "eviccao",
    "enriquecimento sem causa",
    "abuso de direito",
]


class LegislationVectorDB:
    """
    Interface com o banco de dados vetorial ChromaDB e buscador híbrido.
    Persiste os chunks de legislação, realiza buscas híbridas com recuperação
    em duas etapas e aplica reranking jurídico multi-critério.
    """

    def __init__(
        self,
        db_path: str = ".reinan_cache/chromadb",
        collection_name: str = "legislacao",
        embedding_provider: Optional[OllamaEmbeddingProvider] = None,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Inicializa o cliente persistente do ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.db_path))

        # Associa o gerador de embeddings (cria o padrão caso não seja fornecido)
        self.embedding_provider = embedding_provider or OllamaEmbeddingProvider()

        # Obtém ou cria a coleção no ChromaDB
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # Inicializadores para busca lexical
        self._cached_chunks = None
        self._lexical_retriever = None

    def populate(self, chunks: List[Dict[str, Any]], reset: bool = False) -> None:
        """
        Gera os embeddings e salva os chunks de legislação no banco vetorial.

        Args:
            chunks: Lista de dicionários contendo os dados dos artigos.
            reset: Se True, recria/limpa a coleção antes de inserir.
        """
        if reset:
            # Limpa a coleção existente excluindo e recriando
            collection_name = self.collection.name
            try:
                self.client.delete_collection(name=collection_name)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(name=collection_name)

        if not chunks:
            print("[LegislationVectorDB] Lista de chunks vazia. Nada a inserir.")
            return

        # Limpa os caches em memória
        self._cached_chunks = None
        self._lexical_retriever = None

        print(
            f"[LegislationVectorDB] Iniciando a geração de embeddings para {len(chunks)} chunks..."
        )

        # Extrai os textos para gerar embeddings (usando o campo 'text' que contém contextualização)
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_provider.embed_documents(texts)

        # Monta os vetores de dados para inserção no ChromaDB
        ids = []
        documents = []
        metadatas = []

        for idx, chunk in enumerate(chunks):
            # Cria um ID único para cada chunk baseado na lei, artigo e índice
            doc_id = f"{chunk['file_name']}_{chunk['article']}_{idx}"
            ids.append(doc_id)
            documents.append(chunk["text"])

            # Formata palavras-chave como string separada por vírgula para persistência
            keywords_str = ",".join(chunk.get("keywords", []))

            metadatas.append(
                {
                    "article": chunk["article"],
                    "article_number": chunk.get("article_number", ""),
                    "raw_text": chunk.get("raw_text", chunk["text"]),
                    "law_title": chunk["law_title"],
                    "file_name": chunk["file_name"],
                    "legal_area": chunk.get("legal_area", ""),
                    "keywords": keywords_str,
                    "heading_path": chunk.get("heading_path", ""),
                }
            )

        # Insere em lotes (batching) no ChromaDB para evitar limites de carga
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))

            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
            )
            print(f"[LegislationVectorDB] Indexados {end_idx}/{len(ids)} chunks...")

        print(
            f"[LegislationVectorDB] População concluída com sucesso! Total de itens: {self.collection.count()}"
        )

    def _parse_chroma_chunk(self, doc_id: str, doc_text: str, metadata: dict) -> dict:
        """Converte dados retornados do ChromaDB de volta para a estrutura de chunk."""
        meta = metadata or {}
        kws = meta.get("keywords", "")
        keywords_list = kws.split(",") if kws else []

        return {
            "id": doc_id,
            "text": doc_text,
            "raw_text": meta.get("raw_text", doc_text),
            "metadata": meta,
            "keywords": keywords_list,
            "article": meta.get("article", ""),
            "article_number": meta.get("article_number", ""),
            "law_title": meta.get("law_title", ""),
            "file_name": meta.get("file_name", ""),
            "legal_area": meta.get("legal_area", ""),
            "heading_path": meta.get("heading_path", ""),
        }

    def _get_all_chunks(self) -> List[Dict[str, Any]]:
        """Recupera todos os chunks indexados no ChromaDB."""
        if self._cached_chunks is not None:
            return self._cached_chunks

        try:
            res = self.collection.get(include=["documents", "metadatas"])
            chunks = []
            if not (res and "documents" in res and res["documents"]):
                self._cached_chunks = chunks
                return chunks

            docs = res["documents"]
            metas = res["metadatas"]
            ids = res["ids"]
            for i in range(len(docs)):
                chunks.append(self._parse_chroma_chunk(ids[i], docs[i], metas[i]))

            self._cached_chunks = chunks
            return chunks
        except Exception as e:
            print(f"[LegislationVectorDB] Erro ao carregar todos os chunks: {e}")
            return []

    def _ensure_lexical_retriever(self) -> None:
        """Inicializa e treina o buscador lexical caso ainda não esteja pronto."""
        if self._lexical_retriever is not None:
            return

        chunks = self._get_all_chunks()
        self._lexical_retriever = LexicalRetriever()
        self._lexical_retriever.fit(chunks)

    def _find_preferred_model(
        self, preferred_list: list[str], downloaded: list[str]
    ) -> Optional[str]:
        """Procura o primeiro modelo preferido entre os modelos baixados."""
        for preferred in preferred_list:
            for d in downloaded:
                if preferred in d:
                    return d
        return None

    def _get_available_llm_model(self) -> str:
        """Auto-detecta um modelo LLM instalado no Ollama local para reescrita de query."""
        try:
            models_list = self.embedding_provider.client.list()
            downloaded = [m.model for m in models_list.models]
            preferred_models = [
                "gemma2:2b",
                "llama3.2:3b",
                "qwen2.5:3b",
                "gemma2",
                "llama3.2",
                "qwen2.5",
            ]
            match = self._find_preferred_model(preferred_models, downloaded)
            if match:
                return match
            if downloaded:
                return downloaded[0]
        except Exception:
            pass
        return "gemma2:2b"

    # ------------------------------------------------------------------
    # Análise jurídica da questão (heurística, sem LLM)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_literal_legal_text(text: str) -> bool:
        """Detecta se o texto parece conter trecho literal de lei."""
        text_norm = strip_accents(text).lower().strip()
        for pattern in _LEGAL_LITERAL_PATTERNS:
            if re.match(pattern, text_norm):
                return True
        for fragment in _LEGAL_LITERAL_FRAGMENTS:
            if fragment in text_norm:
                return True
        return False

    @staticmethod
    def _extract_legal_branch(q: Dict[str, Any]) -> str:
        """Identifica o ramo provável do Direito a partir da questão."""
        # Usa o campo 'area' se disponível
        area = q.get("area", q.get("category", ""))
        if area:
            return area

        # Heurística por palavras-chave
        statement = q.get("question", q.get("statement", ""))
        choices = q.get("choices", {})
        full_text = strip_accents(statement).lower()
        if choices and "text" in choices:
            full_text += " " + " ".join(
                strip_accents(t).lower() for t in choices["text"]
            )

        best_branch = ""
        best_score = 0
        for branch, keywords in _BRANCH_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in full_text)
            if score > best_score:
                best_score = score
                best_branch = branch

        return best_branch

    @staticmethod
    def _extract_legal_concepts(q: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Extrai instituto jurídico central, verbos, consequências e fundamentos
        presentes na questão e nas alternativas.
        """
        statement = q.get("question", q.get("statement", ""))
        choices = q.get("choices", {})

        full_text = strip_accents(statement).lower()
        choices_text = ""
        if choices and "text" in choices:
            choices_text = " ".join(strip_accents(t).lower() for t in choices["text"])
            full_text += " " + choices_text

        concepts = {
            "verbs": [v for v in _LEGAL_VERBS if v in full_text],
            "consequences": [c for c in _LEGAL_CONSEQUENCES if c in full_text],
            "foundations": [f for f in _LEGAL_FOUNDATIONS if f in full_text],
            "choices_consequences": [
                c for c in _LEGAL_CONSEQUENCES if c in choices_text
            ],
            "choices_foundations": [f for f in _LEGAL_FOUNDATIONS if f in choices_text],
        }
        return concepts

    @staticmethod
    def _extract_distinction_terms(q: Dict[str, Any]) -> List[Tuple[str, str]]:
        """
        Identifica pares de termos que as alternativas distinguem.
        Ex: nulidade vs. anulação, prescrição vs. decadência.
        """
        choices = q.get("choices", {})
        if not choices or "text" not in choices:
            return []

        choice_texts = [strip_accents(t).lower() for t in choices["text"]]

        # Pares de conceitos frequentemente confundidos em questões OAB
        confusion_pairs = [
            ("nulidade", "anulacao"),
            ("nulidade", "anulabilidade"),
            ("nulo", "anulavel"),
            ("prescricao", "decadencia"),
            ("incapacidade absoluta", "incapacidade relativa"),
            ("absolutamente incapaz", "relativamente incapaz"),
            ("erro", "dolo"),
            ("coacao", "estado de perigo"),
            ("causa transitoria", "enfermidade"),
            ("causa transitoria", "deficiencia mental"),
            ("apelacao", "agravo"),
            ("embargos", "reclamacao"),
            ("rescisao", "resolucao"),
            ("dano moral", "dano material"),
        ]

        found_pairs = []
        all_text = " ".join(choice_texts)

        for a, b in confusion_pairs:
            if a in all_text and b in all_text:
                found_pairs.append((a, b))

        return found_pairs

    # ------------------------------------------------------------------
    # Construção de queries (original + reescrita)
    # ------------------------------------------------------------------

    def build_legal_query(self, q: Any, model: Optional[str] = None) -> Tuple[str, str]:
        """
        Gera um par (query_original, query_reescrita) para busca híbrida.

        A query original é usada na busca lexical (captura trechos literais).
        A query reescrita é usada na busca vetorial (captura semântica).

        Returns:
            Tupla (original_query, rewritten_query).
        """
        # Se 'q' já for uma string simples, retorna ela mesma
        if isinstance(q, str):
            return q, q

        statement = q.get("question", q.get("statement", ""))
        choices = q.get("choices", {})
        choices_text = ""
        if choices and "text" in choices:
            choices_text = "\n".join(f"- {t}" for t in choices["text"])

        # Query original: o statement completo (para busca lexical)
        original_query = statement
        if choices and "text" in choices:
            # Adiciona termos das alternativas para busca lexical
            alt_terms = " ".join(
                strip_accents(t).lower().strip(". ")
                for t in choices["text"]
                if len(t) > 3
            )
            original_query = statement + " " + alt_terms

        # Query reescrita via LLM
        rewritten_query = self._build_rewritten_query(
            statement, choices_text, choices, q, model
        )

        return original_query, rewritten_query

    def _clean_rewritten_and_append_choices(self, rewritten: str, choices: Dict) -> str:
        """Limpa a query reescrita e anexa termos normalizados das alternativas."""
        rewritten = re.sub(r"\s+", " ", rewritten)
        rewritten = (
            rewritten.replace("`", "")
            .replace("*", "")
            .replace("Query jurídica gerada:", "")
            .strip()
        )
        if not rewritten:
            return ""

        clean_choices = []
        if choices and "text" in choices:
            for opt in choices["text"]:
                opt_norm = strip_accents(opt).lower().strip(". ")
                if len(opt_norm) > 3:
                    clean_choices.append(opt_norm)
        if clean_choices:
            rewritten += " " + " ".join(clean_choices)
        return rewritten

    def _resolve_llm_model(self, model: Optional[str]) -> str:
        """Resolve qual modelo utilizar para a reescrita de query, evitando modelos de embedding."""
        if model and ("embed" in model.lower() or "nomic" in model.lower()):
            return self._get_available_llm_model()
        return model or self._get_available_llm_model()

    def _build_rewritten_query(
        self,
        statement: str,
        choices_text: str,
        choices: Dict,
        q: Dict,
        model: Optional[str],
    ) -> str:
        """Constrói a query reescrita via LLM ou fallback heurístico."""
        system_prompt = (
            "Você é um assistente especialista em direito brasileiro.\n"
            "Sua tarefa é analisar uma questão de múltipla escolha e extrair uma 'query jurídica' curta e densa para busca (RAG) em códigos de leis.\n"
            "A query jurídica deve:\n"
            "1. Priorizar o ramo do Direito (ex: Direito Civil, Direito Penal, etc.).\n"
            "2. Priorizar os institutos jurídicos centrais envolvidos (ex: estado de perigo, dolo, coação, evicção, erro, nulidade, incapacidade, causa transitória).\n"
            "3. Incluir verbos jurídicos importantes da questão (ex: anular, indenizar, prescrever, executar, impugnar).\n"
            "4. Incluir termos jurídicos chaves presentes nas alternativas.\n"
            "5. Preservar trechos que pareçam ser citação literal de lei (ex: 'São incapazes, relativamente a certos atos').\n"
            "6. Ignorar ou minimizar termos puramente narrativos do caso concreto que não possuem relevância jurídica direta (como 'viagem aérea', 'piscina', 'pizzaria', 'passageiro', 'aeroporto', nomes de pessoas, etc.).\n\n"
            "Retorne APENAS a query jurídica reescrita (uma lista de termos e conceitos separados por espaço), sem explicações, preâmbulos, aspas ou formatação markdown."
        )

        user_prompt = f"Questão: {statement}\n"
        if choices_text:
            user_prompt += f"Alternativas:\n{choices_text}\n"
        user_prompt += "\nQuery jurídica gerada:"

        llm_model = self._resolve_llm_model(model)

        try:
            response = self.embedding_provider.client.chat(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            rewritten_raw = response["message"]["content"].strip()
            rewritten = self._clean_rewritten_and_append_choices(rewritten_raw, choices)
            if rewritten:
                return rewritten
        except Exception as e:
            print(
                f"[RAG] Erro ao chamar LLM para reescrita: {e}. Usando fallback por heurística."
            )

        # Fallback de heurística estática se o LLM falhar
        return self._build_fallback_query(statement, choices, q)

    def _extract_choices_fallback_terms(
        self, choices: Dict, fallback_terms: List[str]
    ) -> None:
        """Extrai termos das alternativas para enriquecer a query de fallback."""
        if not (choices and "text" in choices):
            return
        for opt in choices["text"]:
            opt_norm = strip_accents(opt).lower().strip(". ")
            if len(opt_norm) > 3 and opt_norm not in fallback_terms:
                fallback_terms.append(opt_norm)

    def _build_fallback_query(self, statement: str, choices: Dict, q: Dict) -> str:
        """Constrói uma query de fallback usando heurísticas estáticas."""
        fallback_terms = []
        legal_terms = [
            "estado de perigo",
            "coacao",
            "dolo",
            "erro",
            "lesao",
            "fraude",
            "simulacao",
            "anulabilidade",
            "anulacao",
            "nulidade",
            "nulo",
            "anulavel",
            "incapacidade",
            "causa transitoria",
            "expressao da vontade",
        ]
        text_lower = strip_accents(statement).lower()
        for term in legal_terms:
            if term in text_lower:
                fallback_terms.append(term)

        # Adiciona termos das alternativas diretamente à query de fallback
        self._extract_choices_fallback_terms(choices, fallback_terms)

        # Adiciona área/categoria
        area = q.get("area", q.get("category", ""))
        if area:
            fallback_terms.append(area)

        if not fallback_terms:
            return statement[:150]

        fallback_query = " ".join(fallback_terms)
        print(f"[RAG] Query jurídica de fallback: '{fallback_query}'")
        return fallback_query

    def _process_vector_candidates(
        self, vector_res: Dict[str, Any], candidates: Dict[str, Any]
    ) -> None:
        """Processa candidatos recuperados via similaridade vetorial."""
        if not (vector_res and "documents" in vector_res and vector_res["documents"]):
            return

        docs = vector_res["documents"][0]
        metas = vector_res["metadatas"][0]
        ids = vector_res["ids"][0]
        distances = vector_res.get("distances", [[]])[0]

        for i in range(len(docs)):
            # Normaliza distância L2 para similaridade (0 a 1)
            dist = distances[i] if i < len(distances) else 0.5
            sim_score = 1.0 / (1.0 + dist)
            doc_id = ids[i]
            candidates[doc_id] = {
                "id": doc_id,
                "text": docs[i],
                "metadata": metas[i] or {},
                "vector_score": sim_score,
                "lexical_score": 0.0,
            }

    def _process_lexical_candidates(
        self, lexical_res: List[Dict[str, Any]], candidates: Dict[str, Any]
    ) -> None:
        """Processa candidatos recuperados via busca lexical."""
        for item in lexical_res:
            doc_id = item["id"]
            lex_score = item["lexical_score"]

            if doc_id in candidates:
                candidates[doc_id]["lexical_score"] = lex_score
            else:
                candidates[doc_id] = {
                    "id": doc_id,
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "vector_score": 0.0,
                    "lexical_score": lex_score,
                }

    def query(
        self,
        q: Any,
        top_k: int = 3,
        top_k_retrieval: int = 100,
        top_k_final: Optional[int] = None,
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Pesquisa híbrida em duas etapas com reranking jurídico na base de legislação.
        """
        k_final = top_k_final or top_k

        # Etapa A — Construção das Queries
        original_query, legal_query = self.build_legal_query(q, model)

        # 1. Recuperação Vetorial (ChromaDB) com query reescrita
        query_vector = self.embedding_provider.embed_query(legal_query)
        vector_res = self.collection.query(
            query_embeddings=[query_vector], n_results=top_k_retrieval
        )

        candidates = {}
        self._process_vector_candidates(vector_res, candidates)

        # 2. Recuperação Lexical (BM25) com query original + expandida
        self._ensure_lexical_retriever()
        expanded_original = expand_query(original_query)
        lexical_res = self._lexical_retriever.search(
            original_query, top_k=top_k_retrieval, expanded_query=expanded_original
        )
        self._process_lexical_candidates(lexical_res, candidates)

        # 3. Combinação de Scores (Score Híbrido Base)
        candidate_list = list(candidates.values())
        for c in candidate_list:
            c["base_score"] = 0.5 * c["vector_score"] + 0.5 * c["lexical_score"]

        # 4. Reranking Jurídico Multi-Critério (Etapa B)
        reranked = self.rerank(q, candidate_list)

        # Ordena por score decrescente
        reranked.sort(key=lambda x: x["score"], reverse=True)

        # 5. Avaliação de confiança
        top_results = reranked[:k_final]
        confidence = self._assess_confidence(top_results, q)

        # Anexa informação de confiança a cada resultado
        for r in top_results:
            r["confidence"] = confidence

        return top_results

    # ------------------------------------------------------------------
    # Reranking jurídico multi-critério
    # ------------------------------------------------------------------

    def _count_matches(self, terms: List[str], text: str, min_len: int = 0) -> int:
        """Conta a quantidade de termos que estão presentes no texto."""
        count = 0
        for term in terms:
            if len(term) >= min_len and term in text:
                count += 1
        return count

    def _any_term_in_text(self, terms: List[str], text: str, min_len: int = 0) -> bool:
        """Retorna True se qualquer um dos termos estiver presente no texto."""
        for term in terms:
            if len(term) >= min_len and term in text:
                return True
        return False

    def _find_first_match(self, terms: List[str], text: str) -> Optional[str]:
        """Retorna o primeiro termo da lista que está presente no texto."""
        for term in terms:
            if term in text:
                return term
        return None

    def _check_narrative_penalty(
        self,
        text_norm: str,
        choice_terms_norm: List[str],
        concepts: Dict[str, List[str]],
        narrative_terms: List[str],
    ) -> Tuple[float, List[str]]:
        """Calcula e retorna a penalidade caso o artigo possua apenas termos narrativos."""
        if not narrative_terms:
            return 0.0, []

        matched_narrative = [w for w in narrative_terms if w in text_norm]
        if not matched_narrative:
            return 0.0, []

        has_legal_concept = self._any_term_in_text(
            choice_terms_norm, text_norm, min_len=6
        )
        has_foundation = self._any_term_in_text(
            concepts.get("foundations", []), text_norm
        )
        if not has_legal_concept and not has_foundation:
            pen = min(0.50 * len(matched_narrative), 1.0)
            return pen, [
                f"Apenas narrativa ({matched_narrative[:3]}) sem conceito jurídico (-{pen:.2f})"
            ]

        return 0.0, []

    def _calculate_area_boost(self, q_area_norm: str, chunk_area: str) -> float:
        """Calcula a bonificação por correspondência de área do Direito."""
        if not (q_area_norm and chunk_area):
            return 0.0
        norm_chunk_area = strip_accents(chunk_area).lower()
        if q_area_norm in norm_chunk_area or norm_chunk_area in q_area_norm:
            return 0.30
        return 0.0

    def _calculate_boost(
        self,
        c: Dict[str, Any],
        q_branch: str,
        q_area_norm: str,
        concepts: Dict[str, List[str]],
        choice_terms_norm: List[str],
    ) -> Tuple[float, List[str]]:
        """Calcula o boost total acumulado para um candidato."""
        meta = c["metadata"]
        text_raw = meta.get("raw_text", c["text"])
        text_norm = strip_accents(text_raw).lower()
        heading_path_norm = strip_accents(meta.get("heading_path", "")).lower()
        keywords_str = meta.get("keywords", "")
        chunk_keywords_norm = strip_accents(keywords_str).lower()

        boost = 0.0
        reasons = []

        # A. Correspondência de Ramo Jurídico (+0.30)
        chunk_area = meta.get("legal_area", "")
        area_boost = self._calculate_area_boost(q_area_norm, chunk_area)
        if area_boost > 0.0:
            boost += area_boost
            reasons.append(f"Ramo jurídico compatível: {chunk_area} (+0.30)")

        # B. Instituto jurídico no heading_path (+0.35)
        heading_match = self._find_first_match(
            concepts.get("foundations", []), heading_path_norm
        )
        if heading_match:
            boost += 0.35
            reasons.append(f"Instituto '{heading_match}' no heading_path (+0.35)")

        # C. Consequência jurídica no texto (+0.40)
        consequence_match = self._find_first_match(
            concepts.get("choices_consequences", []), text_norm
        )
        if consequence_match:
            boost += 0.40
            reasons.append(
                f"Consequência '{consequence_match}' encontrada no texto (+0.40)"
            )

        # D. Fundamento jurídico no texto (+0.25)
        foundation_matches = self._count_matches(
            concepts.get("choices_foundations", []), text_norm
        )
        if foundation_matches > 0:
            f_boost = min(0.25 * foundation_matches, 0.50)
            boost += f_boost
            reasons.append(
                f"Fundamentos das alternativas no texto ({foundation_matches}x, +{f_boost:.2f})"
            )

        # E. Verbo jurídico no texto (+0.15)
        verb_match = self._find_first_match(concepts.get("verbs", []), text_norm)
        if verb_match:
            boost += 0.15
            reasons.append(f"Verbo jurídico '{verb_match}' no texto (+0.15)")

        # F. Match parcial de alternativas no texto (+0.20 por match)
        alt_match_count = self._count_matches(choice_terms_norm, text_norm, min_len=6)
        if alt_match_count > 0:
            alt_boost = min(0.20 * alt_match_count, 0.60)
            boost += alt_boost
            reasons.append(
                f"Termos de alternativas no texto ({alt_match_count}x, +{alt_boost:.2f})"
            )

        # G. Correspondência com keywords do chunk (+0.15)
        keyword_match = self._find_first_match(
            concepts.get("foundations", []), chunk_keywords_norm
        )
        if keyword_match:
            boost += 0.15
            reasons.append(f"Keyword '{keyword_match}' nos metadados (+0.15)")

        return boost, reasons

    def _calculate_penalty(
        self,
        c: Dict[str, Any],
        q_branch: str,
        q_area_norm: str,
        concepts: Dict[str, List[str]],
        choice_terms_norm: List[str],
        narrative_terms: List[str],
    ) -> Tuple[float, List[str]]:
        """Calcula a penalidade total para um candidato."""
        meta = c["metadata"]
        text_raw = meta.get("raw_text", c["text"])
        text_norm = strip_accents(text_raw).lower()

        penalty = 0.0
        reasons = []

        chunk_area = meta.get("legal_area", "")
        if q_area_norm and chunk_area:
            norm_chunk_area = strip_accents(chunk_area).lower()
            if self._areas_incompatible(q_area_norm, norm_chunk_area):
                penalty += 0.40
                reasons.append(f"Área incompatível: {chunk_area} vs {q_branch} (-0.40)")

        # H. Penalização: artigo apenas com termos narrativos (-0.50)
        narr_penalty, narr_reasons = self._check_narrative_penalty(
            text_norm, choice_terms_norm, concepts, narrative_terms
        )
        penalty += narr_penalty
        reasons.extend(narr_reasons)

        # I. Penalização: score base muito baixo (-0.20)
        if c["base_score"] < 0.10:
            penalty += 0.20
            reasons.append("Score base muito baixo (<0.10, -0.20)")

        return penalty, reasons

    def rerank(self, q: Any, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aplica reranking jurídico multi-critério genérico e interpretável.
        Usa informações extraídas da questão (ramo, instituto, consequências,
        fundamentos) para pontuar cada candidato.
        """
        # Se 'q' for apenas uma string simples, retorna os scores base sem boosts adicionais
        if isinstance(q, str):
            for c in candidates:
                c["score"] = c["base_score"]
                c["rerank_reason"] = "Score híbrido base (busca vetorial + lexical)."
            return candidates

        # Extrai informações jurídicas da questão
        q_branch = self._extract_legal_branch(q)
        concepts = self._extract_legal_concepts(q)

        q_area_norm = strip_accents(q_branch).lower() if q_branch else ""

        # Termos das alternativas (para match parcial)
        choices = q.get("choices", {})
        choice_terms_norm = []
        if choices and "text" in choices:
            choice_terms_norm = [
                strip_accents(t).lower().strip(". ") for t in choices["text"]
            ]

        # Termos narrativos extraídos dinamicamente do enunciado
        statement = q.get("question", q.get("statement", ""))
        narrative_terms = self._extract_narrative_terms(statement, concepts)

        for c in candidates:
            boost, b_reasons = self._calculate_boost(
                c, q_branch, q_area_norm, concepts, choice_terms_norm
            )
            penalty, p_reasons = self._calculate_penalty(
                c, q_branch, q_area_norm, concepts, choice_terms_norm, narrative_terms
            )

            # Calcula score final
            final_score = c["base_score"] + boost - penalty
            c["score"] = max(0.0, final_score)

            reasons = b_reasons + p_reasons
            # Registra a justificativa do reranking
            c["rerank_reason"] = (
                "; ".join(reasons) if reasons else "Mantido score híbrido base."
            )

            # Detalhamento para exibição no tester.py
            c["boost"] = boost
            c["penalty"] = penalty

        return candidates

    @staticmethod
    def _areas_incompatible(area_a: str, area_b: str) -> bool:
        """Verifica se duas áreas jurídicas são claramente incompatíveis."""
        incompatible_pairs = [
            ("civil", "penal"),
            ("civil", "trabalho"),
            ("civil", "administrativo"),
            ("penal", "trabalho"),
            ("penal", "administrativo"),
            ("penal", "empresarial"),
            ("trabalho", "empresarial"),
            ("trabalho", "administrativo"),
        ]
        for a, b in incompatible_pairs:
            if (a in area_a and b in area_b) or (b in area_a and a in area_b):
                return True
        return False

    @staticmethod
    def _extract_narrative_terms(
        statement: str, concepts: Dict[str, List[str]]
    ) -> List[str]:
        """
        Extrai termos narrativos/cenográficos do enunciado.
        Esses são termos do caso concreto que não são jurídicos.
        """
        # Termos comuns de cenário em questões OAB
        narrative_pool = [
            "viagem",
            "passageiro",
            "piloto",
            "aeroporto",
            "piscina",
            "quadra",
            "pizzaria",
            "pizzas",
            "motociclista",
            "voo",
            "aeronave",
            "surto",
            "remedio",
            "carro",
            "onibus",
            "restaurante",
            "hospital",
            "hotel",
            "loja",
            "shopping",
            "vizinho",
            "amigo",
            "colega",
            "cliente",
            "medico",
            "engenheiro",
            "professor",
            "advogado",
            "contador",
        ]
        text_norm = strip_accents(statement).lower()
        # Filtra apenas os que aparecem no texto e não são termos jurídicos
        all_legal = set()
        for key in concepts:
            all_legal.update(concepts[key])
        return [t for t in narrative_pool if t in text_norm and t not in all_legal]

    # ------------------------------------------------------------------
    # Avaliação de confiança da recuperação
    # ------------------------------------------------------------------

    def _check_key_terms_presence(
        self, results: List[Dict[str, Any]], key_terms: List[str]
    ) -> bool:
        """Verifica se os termos-chave estão presentes em algum dos resultados."""
        if not key_terms:
            return False
        for r in results:
            text_norm = strip_accents(r.get("text", "")).lower()
            if any(t in text_norm for t in key_terms):
                return True
        return False

    def _classify_confidence_level(self, reasons: List[str]) -> Dict[str, Any]:
        """Classifica o nível de confiança com base no número de motivos levantados."""
        if len(reasons) >= 3:
            return {"level": "low", "reason": "; ".join(reasons), "suggested_k": 0}
        if len(reasons) == 2:
            return {"level": "low", "reason": "; ".join(reasons), "suggested_k": 1}
        if len(reasons) == 1:
            return {"level": "medium", "reason": reasons[0], "suggested_k": 2}
        return {
            "level": "high",
            "reason": "Recuperação coerente.",
            "suggested_k": 3,
        }

    def _assess_confidence(
        self, results: List[Dict[str, Any]], q: Any
    ) -> Dict[str, Any]:
        """
        Avalia a qualidade/confiança dos resultados recuperados.
        """
        if not results:
            return {
                "level": "low",
                "reason": "Nenhum resultado recuperado.",
                "suggested_k": 0,
            }

        # 1. Score absoluto do melhor resultado
        top_score = results[0].get("score", 0.0)

        # 2. Spread de scores
        scores = [r.get("score", 0.0) for r in results]
        score_spread = max(scores) - min(scores) if len(scores) > 1 else 0.0

        # 3. Coerência de área jurídica
        areas = [
            r.get("metadata", {}).get("legal_area", "")
            for r in results
            if r.get("metadata", {}).get("legal_area", "")
        ]
        unique_areas = set(strip_accents(a).lower() for a in areas)

        # 4. Presença de termos-chave
        concepts = self._extract_legal_concepts(q) if isinstance(q, dict) else {}
        key_terms = concepts.get("foundations", []) + concepts.get("consequences", [])
        term_found = self._check_key_terms_presence(results, key_terms)

        # Classificação
        reasons = []

        if top_score < 0.20:
            reasons.append(f"Score máximo muito baixo ({top_score:.2f})")
        if len(unique_areas) > 2:
            reasons.append(f"Áreas jurídicas heterogêneas ({len(unique_areas)} áreas)")
        if score_spread < 0.05 and len(results) > 1:
            reasons.append("Scores muito próximos entre resultados de temas diferentes")
        if not term_found and key_terms:
            reasons.append("Nenhum resultado contém termos-chave da questão")

        return self._classify_confidence_level(reasons)

    def count(self) -> int:
        """Retorna a quantidade de registros na coleção."""
        return self.collection.count()
