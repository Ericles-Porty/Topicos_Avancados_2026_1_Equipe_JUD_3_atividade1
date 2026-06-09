import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Any
from bs4 import BeautifulSoup


def strip_accents(text: str) -> str:
    """Remove acentuação de uma string."""
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


FRIENDLY_NAMES = {
    "constituicao": "Constituição Federal de 1988",
    "del2848compilado": "Código Penal (Decreto-Lei 2.848/1940)",
    "del5452compilado": "Consolidação das Leis do Trabalho (CLT)",
    "l10406compilada": "Código Civil (Lei 10.406/2002)",
    "l11101": "Lei de Recuperação Judicial e Falências (Lei 11.101/2005)",
    "l11284": "Lei de Gestão de Florestas Públicas (Lei 11.284/2006)",
    "l13146": "Estatuto da Pessoa com Deficiência (Lei 13.146/2015)",
    "l13869": "Lei de Abuso de Autoridade (Lei 13.869/2019)",
    "l14133": "Nova Lei de Licitações (Lei 14.133/2021)",
    "l5474": "Lei das Duplicatas (Lei 5.474/1968)",
    "l6404consol": "Lei das Sociedades por Ações (Lei 6.404/1976)",
    "l8069compiladoa": "Estatuto da Criança e do Adolescente (ECA)",
    "l8078compilado": "Código de Defesa do Consumidor (CDC)",
    "l8429compilada": "Lei de Improbidade Administrativa (Lei 8.429/1992)",
    "l9279": "Lei da Propriedade Industrial (Lei 9.279/1996)",
}

LEGAL_AREAS = {
    "constituicao": "Direito Constitucional",
    "del2848compilado": "Direito Penal",
    "del5452compilado": "Direito do Trabalho",
    "l10406compilada": "Direito Civil",
    "l11101": "Direito Empresarial",
    "l11284": "Direito Ambiental",
    "l13146": "Direito Civil",
    "l13869": "Direito Penal",
    "l14133": "Direito Administrativo",
    "l5474": "Direito Empresarial",
    "l6404consol": "Direito Empresarial",
    "l8069compiladoa": "Direito da Criança e do Adolescente",
    "l8078compilado": "Código de Defesa do Consumidor (CDC)",
    "l8429compilada": "Lei de Improbidade Administrativa (Lei 8.429/1992)",
    "l9279": "Direito Empresarial",
}

LEGAL_KEYWORDS_POOL = [
    "estado de perigo",
    "coacao",
    "dolo",
    "erro",
    "lesao",
    "fraude contra credores",
    "simulacao",
    "nulidade",
    "anulabilidade",
    "anulacao",
    "contrato",
    "responsabilidade civil",
    "indenizacao",
    "prescricao",
    "decadencia",
    "posse",
    "propriedade",
    "obrigacao",
    "solidariedade",
    "pagamento",
    "mora",
    "perdas e danos",
    "juros",
    "clausula penal",
    "arras",
    "eviccao",
    "vicio redibitorio",
    "alimentos",
    "divorcio",
    "casamento",
    "guarda",
    "adocao",
    "sucessao",
    "heranca",
    "testamento",
    "inventario",
    "partilha",
    "usucapiao",
    "hipoteca",
    "penhor",
    "alienacao fiduciaria",
    "abuso de autoridade",
    "improbidade administrativa",
    "licitacao",
    "consumidor",
    "trabalho",
    "crime",
    "pena",
    "reincidencia",
    "prescricao penal",
    "liberdade",
    "prisao",
    "flagrante",
    "busca e apreensao",
    "processo",
    "contestacao",
    "recurso",
    "apelacao",
    "agravo",
    "monitoria",
    "execucao",
    "cumprimento de sentenca",
    "tutela",
    "liminar",
    "audiencia",
    "pericia",
]


def _build_indexing_text(
    raw_text: str,
    law_title: str,
    current_art: str,
    metadata: dict,
    contextualized_text: str,
) -> str:
    """Constrói o texto de indexação a partir do texto oficial e dos metadados enriquecidos."""
    if not metadata:
        return contextualized_text

    indexing_parts = [f"{law_title} — {current_art}."]

    # Map simple metadata string fields
    field_mappings = [
        ("legal_category", "Categoria"),
        ("canonical_institute", "Instituto"),
        ("legal_effect", "Efeito jurídico"),
    ]
    for key, label in field_mappings:
        val = metadata.get(key)
        if val:
            indexing_parts.append(f"{label}: {val}.")

    # Map list fields
    list_mappings = [
        ("fact_triggers", "Gatilhos fáticos"),
        ("synonyms", "Sinônimos"),
        ("distinguish_from", "Distinguir de"),
        ("synthetic_queries", "Consultas sintéticas"),
    ]
    for key, label in list_mappings:
        vals = metadata.get(key)
        if vals:
            indexing_parts.append(f"{label}: {'; '.join(vals)}.")

    indexing_parts.append(f"Texto oficial: {raw_text}")
    return "\n".join(indexing_parts)


class LegislationChunker:
    """
    Filtra e divide documentos HTML de legislação brasileira em artigos contíguos (chunks).
    Extrai hierarquias estruturais (Parte, Livro, Título, Capítulo, Seção, Subseção)
    e metadados avançados.
    """

    def __init__(self):
        # Padrão para identificar início de artigo no começo de uma linha
        # Exemplos: "Art. 1º", "Art. 10", "art. 1-A", "Artigo 15"
        self.art_pattern = re.compile(
            r"^(art\.\s*\d+|art\s+\d+|artigo\s+\d+)", re.IGNORECASE
        )

    def clean_html_to_lines(self, html_content: str) -> List[str]:
        """
        Remove elementos HTML desnecessários e retorna as linhas do texto normalizado.
        Substitui as quebras de linha existentes por espaços para evitar a quebra
        de termos como 'Art.\n2º' antes de processar os blocos de parágrafos.
        """
        # Normaliza quebras de linha brutas em espaços para evitar corte de palavras
        html_single_line = html_content.replace("\r", " ").replace("\n", " ")

        soup = BeautifulSoup(html_single_line, "html.parser")

        # Deleta tags de metadados, scripts e estilo
        for tag_to_remove in ["script", "style", "head", "title", "meta"]:
            for element in soup.find_all(tag_to_remove):
                element.decompose()

        # Insere quebras de linha após tags de bloco para delimitar os parágrafos corretos
        block_tags = ["p", "div", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br"]
        for tag in block_tags:
            for element in soup.find_all(tag):
                element.insert_after("\n")

        raw_text = soup.get_text()

        # Limpa e filtra as linhas resultantes
        lines = []
        for line in raw_text.splitlines():
            line_clean = line.strip()
            # Normaliza múltiplos espaços em branco
            line_clean = re.sub(r"\s+", " ", line_clean)
            if line_clean:
                lines.append(line_clean)

        return lines

    def _get_heading_level_and_text(
        self, normalized_line: str, raw_line: str
    ) -> tuple[str | None, str | None]:
        """Identifica se uma linha é um cabeçalho estrutural e qual o seu nível."""
        # Se for muito longo, provavelmente é texto corrido
        if len(raw_line) > 150:
            return None, None

        patterns = {
            "subsecao": r"^SUBSECAO\s+(?:[IVXLCDM\d]+|UNICA)\b",
            "secao": r"^SECAO\s+(?:[IVXLCDM\d]+|UNICA)\b",
            "capitulo": r"^CAPITULO\s+(?:[IVXLCDM\d\s]+|UNICO)\b",
            "titulo": r"^TITULO\s+(?:[IVXLCDM\d]+|UNICO)\b",
            "livro": r"^LIVRO\s+(?:[IVXLCDM\d]+)\b",
            "parte": r"^PARTE\s+(?:GERAL|ESPECIAL|[IVXLCDM\d]+)\b",
        }

        for level, pat in patterns.items():
            if re.match(pat, normalized_line):
                return level, raw_line
        return None, None

    def _read_file_content(self, file_path: Path) -> str:
        """Lê o HTML detectando o encoding (UTF-8 com fallback windows-1252).

        UTF-8 é tentado primeiro porque, ao contrário de latin1, ele falha
        em bytes inválidos. Isso permite distinguir arquivos UTF-8 (curadoria
        de Éricles/Fernanda) de arquivos windows-1252/latin1 (curadoria do
        Reinan) sem corromper os acentos de nenhum dos dois.
        """
        data = file_path.read_bytes()
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("windows-1252", errors="replace")

    def _create_chunk(
        self,
        current_chunk: List[str],
        current_art: str,
        current_headings: Dict[str, str],
        law_title: str,
        legal_area: str,
        file_name: str,
    ) -> Dict[str, Any] | None:
        """Gera e retorna a estrutura de dados de um chunk indexado."""
        if not current_chunk or current_art is None:
            return None

        # Constrói o heading_path ativo
        heading_levels = []
        for lvl in ["parte", "livro", "titulo", "capitulo", "secao", "subsecao"]:
            if current_headings.get(lvl):
                heading_levels.append(current_headings[lvl])
        heading_path = " > ".join(heading_levels)

        raw_text = "\n".join(current_chunk)

        # Prepara a contextualização prepended no texto
        prefix_parts = [law_title]
        if heading_path:
            prefix_parts.append(heading_path)
        prefix = f"[{' - '.join(prefix_parts)}] "

        # Texto indexado inclui a hierarquia e o título da lei
        contextualized_text = prefix + raw_text

        # Normalização de número do artigo
        art_number_match = re.search(r"\d+", current_art)
        art_number = art_number_match.group(0) if art_number_match else ""

        # Extração de palavras-chave a partir do texto
        text_norm = strip_accents(raw_text).lower()
        keywords = [kw for kw in LEGAL_KEYWORDS_POOL if kw in text_norm]

        # Process metadata if enriched cache exists
        metadata = {}
        if (
            hasattr(self, "current_enriched_metadata")
            and self.current_enriched_metadata
        ):
            metadata = self.current_enriched_metadata.get(current_art, {})

        indexing_text = _build_indexing_text(
            raw_text, law_title, current_art, metadata, contextualized_text
        )

        return {
            "article": current_art,
            "article_number": art_number,
            "text": contextualized_text,
            "raw_text": raw_text,
            "law_title": law_title,
            "file_name": file_name,
            "legal_area": legal_area,
            "keywords": keywords,
            "heading_path": heading_path,
            "indexing_text": indexing_text,
            "enriched_metadata": metadata,
        }

    def _update_headings(
        self, current_headings: Dict[str, str], level: str, heading_text: str
    ) -> None:
        """Atualiza a estrutura hierárquica e limpa os subníveis correspondentes."""
        current_headings[level] = heading_text
        hierarchy = ["parte", "livro", "titulo", "capitulo", "secao", "subsecao"]
        try:
            idx = hierarchy.index(level)
            # Limpa todos os subníveis que vêm depois do nível atual
            for sub_level in hierarchy[idx + 1 :]:
                current_headings[sub_level] = ""
        except ValueError:
            pass

    def _process_line_for_chunk(
        self,
        line: str,
        state: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        law_title: str,
        legal_area: str,
        file_name: str,
    ) -> None:
        """Processa uma única linha para identificar transições de artigos ou cabeçalhos."""
        line_clean = line.strip()
        line_norm = strip_accents(line_clean).upper()

        # 1. Verifica se a linha é um cabeçalho estrutural
        level, heading_text = self._get_heading_level_and_text(line_norm, line_clean)
        if level:
            chunk_data = self._create_chunk(
                state["current_chunk"],
                state["current_art"],
                state["current_headings"],
                law_title,
                legal_area,
                file_name,
            )
            if chunk_data:
                chunks.append(chunk_data)

            self._update_headings(state["current_headings"], level, heading_text)
            state["current_art"] = None
            state["current_chunk"] = []
            return

        # 2. Verifica se a linha indica o início de um novo artigo
        art_match = self.art_pattern.match(line_clean)
        if art_match:
            chunk_data = self._create_chunk(
                state["current_chunk"],
                state["current_art"],
                state["current_headings"],
                law_title,
                legal_area,
                file_name,
            )
            if chunk_data:
                chunks.append(chunk_data)

            state["current_art"] = re.sub(r"\s+", " ", art_match.group(1).strip())
            state["current_chunk"] = [line_clean]
        else:
            if state["current_art"] is not None:
                state["current_chunk"].append(line_clean)

    def chunk_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Lê o arquivo HTML, limpa o conteúdo e divide em artigos.
        Retorna uma lista de dicionários contendo o artigo, texto e metadados.
        """
        import json

        content = self._read_file_content(file_path)
        lines = self.clean_html_to_lines(content)

        # Normalização do nome da lei
        stem_norm = strip_accents(file_path.stem).lower()
        law_title = FRIENDLY_NAMES.get(stem_norm, file_path.stem)
        legal_area = LEGAL_AREAS.get(stem_norm, "Direito")

        # Load enriched metadata cache if exists
        enriched_path = Path("database/enriched_metadata") / f"{file_path.stem}.json"
        self.current_enriched_metadata = {}
        if enriched_path.exists():
            try:
                with open(enriched_path, "r", encoding="utf-8") as f:
                    self.current_enriched_metadata = json.load(f)
            except Exception as e:
                print(f"[LegislationChunker] Erro ao carregar metadados ricos: {e}")

        state = {
            "current_headings": {
                "parte": "",
                "livro": "",
                "titulo": "",
                "capitulo": "",
                "secao": "",
                "subsecao": "",
            },
            "current_chunk": [],
            "current_art": "Preâmbulo",
        }

        chunks = []
        for line in lines:
            self._process_line_for_chunk(
                line, state, chunks, law_title, legal_area, file_path.name
            )

        # Salva o último artigo após o loop
        chunk_data = self._create_chunk(
            state["current_chunk"],
            state["current_art"],
            state["current_headings"],
            law_title,
            legal_area,
            file_path.name,
        )
        if chunk_data:
            chunks.append(chunk_data)

        # Clear state after run
        self.current_enriched_metadata = {}

        return chunks
