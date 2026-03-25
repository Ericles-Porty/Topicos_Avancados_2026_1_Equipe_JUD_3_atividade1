import json
import csv
from pathlib import Path
from typing import List, Dict, Any

class StorageManager:
    """
    Gerencia o armazenamento local de arquivos em um diretório de cache reservado.
    """
    def __init__(self, cache_dir: str = ".reinan_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def save_data(self, data: List[Dict[str, Any]], filename: str, fmt: str = "json", sub_dir: str = None) -> Path:
        """
        Salva a lista de dicionários num arquivo dentro da pasta de cache, com o formato escolhido.
        """
        target_dir = self.cache_dir
        if sub_dir:
            target_dir = self.cache_dir / sub_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            
        filepath = target_dir / f"{filename}.{fmt}"
        
        if fmt == "json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        elif fmt == "csv":
            if not data:
                return filepath
            keys = data[0].keys()
            with open(filepath, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(data)
        else:
            raise ValueError(f"Formato de arquivo não suportado: {fmt}")
            
        return filepath

    def load_data(self, filename: str, fmt: str = "json", sub_dir: str = None) -> List[Dict[str, Any]]:
        """
        Carrega dados de formato JSON ou CSV diretamente do diretório de cache.
        """
        target_dir = self.cache_dir
        if sub_dir:
            target_dir = self.cache_dir / sub_dir
            
        filepath = target_dir / f"{filename}.{fmt}"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Arquivo não encontrado no diretório: {filepath}")
            
        if fmt == "json":
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        elif fmt == "csv":
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                return [dict(row) for row in reader]
        else:
            raise ValueError(f"Formato de arquivo não suportado: {fmt}")

    def list_available_models(self, dataset_name: str, sub_dir: str = "results") -> List[str]:
        """
        Retorna a lista de nomes de modelos que possuem resultados salvos em cache 
        para um determinado dataset.
        """
        target_dir = self.cache_dir / sub_dir
        if not target_dir.exists():
            return []
            
        models = []
        prefix = f"{dataset_name}_"
        suffix = "_results.json"
        
        for file in target_dir.glob(f"{prefix}*{suffix}"):
            model_name_dashed = file.name[len(prefix):-len(suffix)]
            model_name = model_name_dashed.replace('-', ':')
            models.append(model_name)
            
        return models
