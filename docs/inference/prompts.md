# Prompts de inferencia

## Motor de templates

O projeto utiliza **MiniJinja** para renderizar prompts dinamicamente. Os templates estao em `src/templates/`.

## Questoes abertas

### Estrutura da mensagem

As questoes abertas utilizam o **system prompt fornecido pelo dataset** (campo `system`), que define o papel do candidato e as regras da prova:

```
messages = [
    {"role": "system", "content": <system do dataset>},
    {"role": "user",   "content": <enunciado>},
    {"role": "user",   "content": <turn_1>},   # se existir
    {"role": "user",   "content": <turn_2>},   # se existir
    ...
]
```

Cada subitem (`turn`) e adicionado como uma mensagem separada do usuario, permitindo que o modelo processe cada subpergunta no contexto da conversa.

## Multipla escolha

### System prompt

**Arquivo:** `src/templates/multiple_choice_system.jinja`

O system prompt instrui o modelo a retornar **exclusivamente um JSON** com a alternativa escolhida:

```
Escolha a alternativa correta referente a questao.
ATENCAO: Nao adicione justificativas, pensamentos, textos complementares ou marcacoes de markdown.
Voce deve retornar ESTRITAMENTE um objeto JSON valido:

{"resposta": "letra da alternativa"}
```

### User prompt

**Arquivo:** `src/templates/multiple_choice.jinja`

O user prompt formata o enunciado seguido das alternativas:

```
{{ question }}

A) {{ texto_a }}
B) {{ texto_b }}
C) {{ texto_c }}
D) {{ texto_d }}
```

### Extracao da resposta

Na avaliacao, a resposta e extraida em duas etapas:
1. **Tentativa JSON:** Tenta parsear a resposta como JSON e extrair o campo `"resposta"`
2. **Fallback regex:** Se o JSON falhar, busca a primeira letra A-D no texto com regex

Esta abordagem dupla garante robustez mesmo quando o modelo nao segue o formato JSON solicitado.
