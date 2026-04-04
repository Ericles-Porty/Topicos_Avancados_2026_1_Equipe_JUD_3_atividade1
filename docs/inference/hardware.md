# Hardware

## Configuracao utilizada

Os experimentos de inferencia foram executados em uma maquina local com a seguinte configuracao:

| Componente | Especificacao |
|---|---|
| **GPU** | NVIDIA GeForce GTX 1650 |
| **VRAM dedicada** | 4,0 GB |
| **RAM** | 16 GB |
| **SO** | Windows 11 Pro |
| **Runtime** | Ollama (execucao local) |

## Consideracoes

- Todos os modelos selecionados cabem confortavelmente nos 4 GB de VRAM
- A inferencia e executada inteiramente na GPU, sem necessidade de offloading para RAM
- O Ollama gerencia automaticamente o carregamento e descarregamento dos modelos
- Apenas um modelo e carregado em memoria por vez durante a inferencia
