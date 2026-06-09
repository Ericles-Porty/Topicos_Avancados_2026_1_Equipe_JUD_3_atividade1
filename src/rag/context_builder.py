"""
Construtor de contexto RAG.

Porta a lógica de montagem do bloco [LEGISLAÇÃO RELEVANTE] e dos alertas de
distinção que, no repositório original do Reinan, residia na classe
ExecutionManager (src/execution/base.py). Aqui ela é isolada em uma classe
reutilizável que recebe uma instância de LegislationVectorDB e devolve o
contexto textual compacto e as informações estruturadas de recuperação
(rag_info), aplicando a porta de confiança e os alertas de distinção.
"""

from typing import Any, List, Optional

from rag.chunker import strip_accents


_INSTITUTE_DEFINITIONS = {
    "erro": "falsa percepção espontânea da realidade.",
    "dolo": "induzimento malicioso ao erro por parte de outrem.",
    "coacao": "ameaça grave e iminente que infunde temor.",
    "estado de perigo": "emergência/grave dano e obrigação excessivamente onerosa.",
    "lesao": "obrigação desproporcional por necessidade premente ou inexperiência.",
    "fraude": "alienação de bens para frustrar execução ou cobrança.",
    "simulacao": "declaração enganosa para aparentar negócio inexistente ou diferente.",
    "nulidade": "invalidade absoluta e insanável por infração de ordem pública.",
    "anulabilidade": "invalidade relativa, sanável ou confirmável pelas partes.",
    "prescricao": "extinção da pretensão de cobrar/exigir um direito pelo decurso do tempo.",
    "decadencia": "extinção do próprio direito potestativo por falta de exercício no prazo.",
    "incapacidade absoluta": "invalidade absoluta do ato (ex: menores de 16 anos).",
    "incapacidade relativa": "invalidade relativa (ex: maiores de 16 e menores de 18 ou causa transitória).",
    "causa transitoria": "impedimento temporário de exprimir a vontade, gerando incapacidade relativa.",
}


class RagContextBuilder:
    """
    Realiza a busca híbrida no banco vetorial e monta o contexto textual
    (e info estruturada) a ser injetado no prompt do LLM.
    """

    def __init__(self, db, top_k: int = 3):
        self.db = db
        self.top_k = top_k

    def _resolve_retrieval_notes(self, meta: dict) -> str:
        """Resolve a descrição interpretativa (Uso) a partir dos metadados ricos."""
        retrieval_notes = meta.get("retrieval_notes", "")
        if retrieval_notes:
            return retrieval_notes

        institute = meta.get("canonical_institute", "")
        category = meta.get("legal_category", "")
        effect = meta.get("legal_effect", "")
        if not institute:
            return f"dispositivo legal relativo a {meta.get('legal_area', 'Direito')}."

        notes = f"define o instituto '{institute}'"
        if category:
            notes += f" no âmbito de '{category}'"
        if effect:
            notes += f", com o efeito jurídico de '{effect}'"
        return notes

    def _process_single_rag_result(
        self, idx: int, res: dict, confidence_level: str
    ) -> tuple[dict, str]:
        """Processa um único resultado RAG e retorna info estruturada e formatada."""
        meta = res.get("metadata", {})
        law_title = meta.get("law_title", "Desconhecida")
        file_name = meta.get("file_name", "")
        article = meta.get("article", "Desconhecido")
        score = res.get("score", 0.0)
        raw_text = meta.get("raw_text", res.get("text", ""))

        law_str = f"{law_title}"
        if file_name:
            law_str += f" ({file_name})"

        info = {
            "Lei": law_str,
            "Artigo": article,
            "Score": round(score, 4),
            "Score Vetorial Base": round(res.get("vector_score", 0.0), 4),
            "Score Lexical Base": round(res.get("lexical_score", 0.0), 4),
            "Score Hibrido Base": round(res.get("base_score", 0.0), 4),
            "Boost": round(res.get("boost", 0.0), 4),
            "Penalidade": round(res.get("penalty", 0.0), 4),
            "Justificativa": res.get("rerank_reason", ""),
            "Confianca": confidence_level,
        }

        retrieval_notes = self._resolve_retrieval_notes(meta)

        indented_text = "\n".join(f"   {line}" for line in raw_text.strip().split("\n"))

        context_part = (
            f"{idx + 1}. {law_title} — {article}\n"
            f"   Uso: {retrieval_notes}\n"
            f"   Texto:\n"
            f"{indented_text}"
        )
        return info, context_part

    def _query_rag(self, q: Any, k: int, model: Optional[str]) -> List[dict]:
        """Consulta o banco vetorial RAG."""
        try:
            return self.db.query(q, top_k=k, top_k_retrieval=100, model=model) or []
        except Exception as e:
            print(f"[RAG] Erro ao consultar banco vetorial: {e}")
            return []

    def _apply_confidence_threshold(
        self, results: List[dict], k: int
    ) -> tuple[str, List[dict], Optional[dict]]:
        """Aplica a avaliação de confiança para filtrar os resultados."""
        if not results:
            return "high", [], None

        confidence = results[0].get("confidence", {})
        confidence_level = confidence.get("level", "high")
        suggested_k = confidence.get("suggested_k", k)
        effective_k = min(len(results), suggested_k)

        if effective_k <= 0:
            print(
                f"[RAG] Confiança baixa ({confidence.get('reason', '')}). Fallback sem RAG."
            )
            return confidence_level, [], {"confidence": confidence}

        return confidence_level, results[:effective_k], None

    def _collect_single_meta_concepts(self, res: dict, concepts: set) -> None:
        """Extrai institutos e distinções dos metadados de um único resultado RAG."""
        meta = res.get("metadata", {})
        inst = meta.get("canonical_institute")
        if inst:
            concepts.add(inst.strip().lower())
        dist = meta.get("distinguish_from", [])
        if isinstance(dist, list):
            for d in dist:
                if d:
                    concepts.add(d.strip().lower())

    def _collect_meta_concepts(self, results: List[dict]) -> set:
        """Extrai institutos e distinções dos metadados dos resultados RAG."""
        concepts = set()
        for res in results:
            self._collect_single_meta_concepts(res, concepts)
        return concepts

    def _collect_choice_concepts(self, q: Any) -> set:
        """Extrai conceitos das alternativas da questão."""
        concepts = set()
        if not isinstance(q, dict):
            return concepts

        choices = q.get("choices", {})
        if not (choices and "text" in choices):
            return concepts

        for choice_text in choices["text"]:
            choice_clean = strip_accents(choice_text).lower().strip(". ")
            for key in _INSTITUTE_DEFINITIONS:
                if key in choice_clean:
                    concepts.add(key)
        return concepts

    def _collect_distinct_concepts(self, results: List[dict], q: Any) -> set:
        """Coleta conceitos jurídicos dos metadados e das alternativas da questão."""
        meta_concepts = self._collect_meta_concepts(results)
        choice_concepts = self._collect_choice_concepts(q)
        return meta_concepts.union(choice_concepts)

    def _compile_alerts(
        self, q: Any, distinct_concepts: set, confidence_level: str
    ) -> str:
        """Gera alertas de distinção (estáticos e dinâmicos) e de nível de confiança."""
        dynamic_alerts = []
        for concept in sorted(distinct_concepts):
            if concept in _INSTITUTE_DEFINITIONS:
                dynamic_alerts.append(
                    f"{concept.capitalize()} envolve {_INSTITUTE_DEFINITIONS[concept]}"
                )

        distinction_alerts = self._build_distinction_alerts(q)

        for alert in dynamic_alerts:
            concept_name = alert.split(" ")[0].lower()
            if not any(
                concept_name in static_alert.lower()
                for static_alert in distinction_alerts
            ):
                distinction_alerts.append(alert)

        alert_parts = []
        if distinction_alerts:
            alert_parts.append(
                "\n\n[ALERTA DE DISTINÇÃO]\n"
                + "\n".join(f"- {alert}" for alert in distinction_alerts)
            )

        if confidence_level != "high":
            alert_parts.append(
                f"\n\n[ALERTA] Confiança da recuperação: {confidence_level}. Use os artigos com cautela."
            )

        return "".join(alert_parts)

    def get_context_and_info(
        self, q: Any, top_k: Optional[int] = None, model: Optional[str] = None
    ) -> tuple[str, list]:
        """
        Realiza busca híbrida no banco vetorial e retorna contexto textual
        compacto e info estruturada. Aplica avaliação de confiança para ajustar
        o volume de contexto enviado ao modelo.
        """
        if not self.db:
            return "", []

        k = top_k if top_k is not None else self.top_k
        results = self._query_rag(q, k, model)
        if not results:
            return "", []

        confidence_level, results, fallback_res = self._apply_confidence_threshold(
            results, k
        )
        if fallback_res is not None:
            return "", [fallback_res]

        rag_info = []
        context_parts = []
        for idx, res in enumerate(results):
            info, context_part = self._process_single_rag_result(
                idx, res, confidence_level
            )
            rag_info.append(info)
            context_parts.append(context_part)

        context_str = "[LEGISLAÇÃO RELEVANTE]\n" + "\n\n".join(context_parts)
        distinct_concepts = self._collect_distinct_concepts(results, q)
        context_str += self._compile_alerts(q, distinct_concepts, confidence_level)

        return context_str, rag_info

    @staticmethod
    def _build_distinction_alerts(q: Any) -> list:
        """
        Gera alertas curtos de distinção jurídica com base nos pares de conceitos
        que as alternativas distinguem. Ajuda o modelo pequeno a não confundir.
        """
        if not isinstance(q, dict):
            return []

        try:
            from rag.database import LegislationVectorDB

            pairs = LegislationVectorDB._extract_distinction_terms(q)
        except Exception:
            return []

        if not pairs:
            return []

        pair_explanations = {
            (
                "nulidade",
                "anulacao",
            ): "Nulidade = invalidade absoluta (Art. 166). Anulação = invalidade relativa (Art. 171). Se o fundamento apontar para 'anulável', NÃO escolha 'nulidade'.",
            (
                "nulidade",
                "anulabilidade",
            ): "Nulidade = ato nulo de pleno direito. Anulabilidade = ato anulável por vício sanável.",
            (
                "nulo",
                "anulavel",
            ): "Ato nulo não produz efeitos. Ato anulável produz efeitos até ser anulado.",
            (
                "prescricao",
                "decadencia",
            ): "Prescrição extingue a pretensão (direito subjetivo). Decadência extingue o próprio direito (potestativo).",
            (
                "incapacidade absoluta",
                "incapacidade relativa",
            ): "Absolutamente incapaz = menores de 16 anos. Relativamente incapaz = maiores de 16 e menores de 18 ou com causa transitória.",
            (
                "absolutamente incapaz",
                "relativamente incapaz",
            ): "Absolutamente incapaz → ato NULO. Relativamente incapaz → ato ANULÁVEL.",
            (
                "erro",
                "dolo",
            ): "Erro = falsa percepção espontânea. Dolo = indução em erro pela outra parte.",
            (
                "coacao",
                "estado de perigo",
            ): "Coação = ameaça. Estado de perigo = necessidade de salvar a si ou parente.",
            (
                "causa transitoria",
                "enfermidade",
            ): "Causa transitória (Art. 4.º, III) = incapacidade relativa → anulação. Enfermidade/deficiência mental não é mais causa automática de incapacidade (Lei 13.146/2015).",
            (
                "causa transitoria",
                "deficiencia mental",
            ): "Causa transitória = impedimento temporário de exprimir vontade → incapacidade relativa. Deficiência mental = não afeta automaticamente a capacidade (Lei 13.146/2015).",
            (
                "apelacao",
                "agravo",
            ): "Apelação = recurso contra sentença. Agravo de instrumento = recurso contra decisão interlocutória.",
            (
                "rescisao",
                "resolucao",
            ): "Rescisão = término de contrato por causa superveniente. Resolução = término por inadimplemento.",
            (
                "dano moral",
                "dano material",
            ): "Dano moral = lesão a direito de personalidade. Dano material = prejuízo patrimonial efetivo.",
        }

        alerts = []
        for pair in pairs:
            explanation = pair_explanations.get(pair) or pair_explanations.get(
                (pair[1], pair[0])
            )
            if explanation:
                alerts.append(explanation)
            else:
                alerts.append(f"Atenção à diferença entre '{pair[0]}' e '{pair[1]}'.")

        return alerts
