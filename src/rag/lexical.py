import re
import math
import unicodedata
from collections import Counter
from typing import List, Dict, Any, Optional


def strip_accents(text: str) -> str:
    """Remove acentuação de uma string."""
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


# ---------------------------------------------------------------------------
# Dicionário de Expansão Jurídica
# Cada chave é um termo canônico; os valores são variantes e sinônimos.
# A expansão é bidirecional: qualquer variante também mapeia para todas as
# outras variantes do mesmo grupo.
# ---------------------------------------------------------------------------
LEGAL_EXPANSION: Dict[str, List[str]] = {
    # Invalidade dos negócios jurídicos
    "anulacao": ["anulavel", "anulabilidade", "invalidade relativa", "anular"],
    "nulidade": ["nulo", "invalidade absoluta", "nulo de pleno direito"],
    # Capacidade civil
    "incapacidade": [
        "incapaz",
        "incapazes",
        "relativamente incapaz",
        "absolutamente incapaz",
        "incapacidade relativa",
        "incapacidade absoluta",
    ],
    "expressao da vontade": [
        "exprimir vontade",
        "exprimir sua vontade",
        "manifestar vontade",
        "discernimento",
        "causa transitoria",
    ],
    # Vícios do consentimento
    "estado de perigo": ["perigo iminente", "prestacao excessiva"],
    "coacao": ["coagir", "ameaca", "temor"],
    "dolo": ["induzir em erro", "artificio", "ardil"],
    "erro": ["erro substancial", "falsa representacao", "erro de direito"],
    "lesao": ["prestacao desproporcional", "necessidade", "inexperiencia"],
    "fraude contra credores": [
        "fraude",
        "pauliana",
        "acao revocatoria",
        "insolvencia",
    ],
    "simulacao": ["negocio simulado", "dissimulacao"],
    # Prazos
    "prescricao": [
        "prescreve",
        "prazo prescricional",
        "pretensao",
        "prescricao extintiva",
    ],
    "decadencia": ["prazo decadencial", "decai", "direito potestativo"],
    # Responsabilidade civil
    "responsabilidade civil": [
        "dano moral",
        "dano material",
        "indenizacao",
        "reparacao",
    ],
    "indenizacao": ["indenizar", "perdas e danos", "reparar"],
    # Contratos
    "eviccao": ["evicto", "garantia", "perda da coisa"],
    "vicio redibitorio": ["defeito oculto", "coisa defeituosa"],
    # Direitos reais
    "usucapiao": ["posse prolongada", "aquisicao originaria"],
    "propriedade": ["dominio", "proprietario", "titularidade"],
    # Família e sucessões
    "divorcio": ["dissolucao do casamento", "separacao"],
    "alimentos": ["pensao alimenticia", "obrigacao alimentar"],
    "guarda": ["guarda compartilhada", "guarda unilateral"],
    "heranca": ["herdeiro", "sucessao", "espolio"],
    "testamento": ["disposicao testamentaria", "legado", "testador"],
    # Processo civil
    "recurso": ["apelacao", "agravo", "embargos", "reclamacao"],
    "apelacao": ["recurso de apelacao", "segundo grau"],
    "agravo": ["agravo de instrumento", "agravo interno"],
    "tutela": ["tutela antecipada", "tutela de urgencia", "tutela provisoria"],
    "execucao": ["penhora", "adjudicacao", "bloqueio", "expropriacao"],
    # Direito penal
    "crime": ["delito", "infração penal", "tipo penal"],
    "pena": ["sancao penal", "reprimenda", "apenado"],
    # Direito do consumidor
    "consumidor": ["fornecedor", "relacao de consumo", "produto defeituoso"],
    # Direito administrativo
    "licitacao": ["pregao", "concorrencia", "tomada de precos", "convite"],
    "improbidade": [
        "ato de improbidade",
        "enriquecimento ilicito",
        "dano ao erario",
    ],
    # Direito constitucional
    "mandado de seguranca": ["direito liquido e certo", "autoridade coatora"],
    "habeas corpus": ["liberdade de locomocao", "constrangimento ilegal"],
    # Direito trabalhista
    "rescisao": ["demissao", "dispensa", "justa causa", "verbas rescisorias"],
    "estabilidade": ["garantia de emprego", "estabilidade provisoria"],
    # Direito empresarial
    "falencia": ["massa falida", "credores", "ativo"],
    "recuperacao judicial": ["plano de recuperacao", "assembleia de credores"],
    # Pessoa com deficiência
    "deficiencia": [
        "pessoa com deficiencia",
        "capacidade civil",
        "plena capacidade",
        "estatuto da pessoa com deficiencia",
    ],
}


def _build_expansion_index() -> Dict[str, List[str]]:
    """
    Constrói um índice invertido onde cada variante/termo aponta para
    todas as outras variantes do mesmo grupo (incluindo o termo canônico).
    """
    index: Dict[str, List[str]] = {}
    for canonical, variants in LEGAL_EXPANSION.items():
        all_terms = [canonical] + variants
        for term in all_terms:
            norm = strip_accents(term).lower().strip()
            peers = [strip_accents(t).lower().strip() for t in all_terms if t != term]
            if norm in index:
                existing = set(index[norm])
                existing.update(peers)
                index[norm] = list(existing)
            else:
                index[norm] = peers
    return index


_EXPANSION_INDEX = _build_expansion_index()


def _expand_ngrams_for_n(
    n: int,
    words: List[str],
    matched_spans: set,
    expansions: set,
) -> None:
    """Procura e registra expansões para n-grams de tamanho n."""
    for i in range(len(words) - n + 1):
        # Pula se qualquer posição já foi matched por n-gram maior
        if any(j in matched_spans for j in range(i, i + n)):
            continue
        ngram = " ".join(words[i : i + n])
        if ngram in _EXPANSION_INDEX:
            expansions.update(_EXPANSION_INDEX[ngram])
            matched_spans.update(range(i, i + n))


def expand_query(query: str) -> str:
    """
    Expande uma query com sinônimos jurídicos do dicionário.
    Primeiro tenta casar n-grams longos (3, 2 palavras), depois unigramas.
    Retorna a query original com os termos expandidos anexados.
    """
    query_norm = strip_accents(query).lower()
    words = query_norm.split()
    expansions = set()
    matched_spans = set()  # Para evitar duplicação

    for n in [3, 2, 1]:
        _expand_ngrams_for_n(n, words, matched_spans, expansions)

    if not expansions:
        return query

    return query + " " + " ".join(expansions)


# ---------------------------------------------------------------------------
# BM25 — Implementação manual leve (sem dependências externas)
# ---------------------------------------------------------------------------


class BM25Retriever:
    """
    Implementação manual de BM25 (Okapi BM25) para busca lexical em chunks
    de legislação. Suporta n-grams (1,2) e normalização de acentos.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_dl = 0.0
        self.doc_freqs: Dict[str, int] = {}  # term -> nº de docs que contêm o term
        self.doc_lengths: List[int] = []
        self.term_freqs: List[Dict[str, int]] = []  # por documento
        self.chunks: List[Dict[str, Any]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Tokeniza texto em unigramas e bigramas, com normalização."""
        text_norm = strip_accents(text).lower()
        # Remove pontuação e caracteres especiais
        text_norm = re.sub(r"[^\w\s]", " ", text_norm)
        words = [w for w in text_norm.split() if len(w) >= 2]

        tokens = list(words)  # unigramas
        # bigramas
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]} {words[i + 1]}")

        return tokens

    def fit(self, chunks: List[Dict[str, Any]]) -> None:
        """Indexa os chunks para busca BM25."""
        self.chunks = chunks
        self.corpus_size = len(chunks)
        self.doc_freqs = {}
        self.doc_lengths = []
        self.term_freqs = []

        if not chunks:
            return

        for chunk in chunks:
            text = chunk.get("text", "")
            tokens = self._tokenize(text)
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            self.doc_lengths.append(len(tokens))

            # Conta document frequency (cada term aparece no máximo 1x por doc)
            for term in set(tokens):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        total_tokens = sum(self.doc_lengths)
        self.avg_dl = total_tokens / self.corpus_size if self.corpus_size > 0 else 0.0

    def _score_document(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calcula o score BM25 de um documento para os tokens da query."""
        score = 0.0
        tf_doc = self.term_freqs[doc_idx]
        dl = self.doc_lengths[doc_idx]

        for term in query_tokens:
            if term not in self.doc_freqs:
                continue

            # IDF com suavização (IDF nunca negativo)
            df = self.doc_freqs[term]
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

            # TF normalizado com BM25
            freq = tf_doc.get(term, 0)
            if freq == 0:
                continue

            tf_norm = (freq * (self.k1 + 1)) / (
                freq + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            )

            score += idf * tf_norm

        return score

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Busca os chunks mais relevantes para a query via BM25."""
        if not self.chunks or self.corpus_size == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Calcula score para todos os documentos
        scores = []
        for idx in range(self.corpus_size):
            s = self._score_document(query_tokens, idx)
            if s > 0.0:
                scores.append((idx, s))

        # Ordena por score decrescente
        scores.sort(key=lambda x: x[1], reverse=True)

        # Normaliza scores para [0, 1] relativo ao maior score
        max_score = scores[0][1] if scores else 1.0

        results = []
        for idx, raw_score in scores[:top_k]:
            chunk_copy = self.chunks[idx].copy()
            chunk_copy["lexical_score"] = (
                raw_score / max_score if max_score > 0 else 0.0
            )
            results.append(chunk_copy)

        return results


class LexicalRetriever:
    """
    Realiza busca lexical BM25 sobre os chunks de legislação com expansão
    jurídica de termos e busca dupla (query original + query expandida).
    """

    def __init__(self):
        self.bm25 = BM25Retriever()
        self.chunks: List[Dict[str, Any]] = []

    def fit(self, chunks: List[Dict[str, Any]]) -> None:
        """Indexa os chunks no motor BM25."""
        self.chunks = chunks
        self.bm25.fit(chunks)

    def search(
        self,
        query: str,
        top_k: int = 20,
        expanded_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Busca híbrida lexical: executa BM25 com a query original e com a
        query expandida juridicamente, fundindo os resultados.
        """
        if not self.chunks:
            return []

        # Busca 1: query original (captura trechos literais de lei)
        results_original = self.bm25.search(query, top_k=top_k)

        # Busca 2: query expandida com sinônimos jurídicos
        if expanded_query is None:
            expanded_query = expand_query(query)

        if expanded_query != query:
            results_expanded = self.bm25.search(expanded_query, top_k=top_k)
        else:
            results_expanded = []

        # Fusão: combina scores dando peso 0.6 para original e 0.4 para expandida
        score_map: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        for item in results_original:
            doc_id = item["id"]
            score_map[doc_id] = 0.6 * item["lexical_score"]
            chunk_map[doc_id] = item

        for item in results_expanded:
            doc_id = item["id"]
            expanded_contribution = 0.4 * item["lexical_score"]
            if doc_id in score_map:
                score_map[doc_id] += expanded_contribution
            else:
                score_map[doc_id] = expanded_contribution
                chunk_map[doc_id] = item

        # Ordena por score fusionado
        sorted_ids = sorted(score_map, key=score_map.get, reverse=True)

        results = []
        for doc_id in sorted_ids[:top_k]:
            chunk_copy = chunk_map[doc_id].copy()
            chunk_copy["lexical_score"] = score_map[doc_id]
            results.append(chunk_copy)

        return results
