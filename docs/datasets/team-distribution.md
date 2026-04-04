# Distribuicao de questoes

## Criterio de distribuicao

As questoes dos datasets foram divididas entre os membros da equipe em intervalos contiguos, garantindo que cada membro trabalhe com um subconjunto exclusivo e complementar.

## Tabela de distribuicao

| Membro | Questoes Abertas (J1) | Multipla Escolha (J2) |
|---|---|---|
| Fernanda Mirely | 141 a 152 | 1477 a 1599 |
| **Ericles** | **153 a 164** | **1600 a 1722** |
| Julia | 165 a 176 | 1723 a 1845 |
| Reinan | 177 a 188 | 1846 a 1968 |
| Mikaela | 189 a 200 | 1969 a 2091 |
| Victor Leonardo | 201 a 210 | 2092 a 2210 |

## Subconjunto deste repositorio

Este repositorio processa as questoes designadas para **Ericles**:

| Dataset | Intervalo | Indices Python (base zero) | Quantidade |
|---|---|---|---|
| **OAB Bench** | 153 a 164 | `slice(153, 165)` | 12 questoes |
| **OAB Exams** | 1600 a 1722 | `slice(1600, 1723)` | 123 questoes |

### Implementacao no codigo

No arquivo `src/load_dataset.py`, os intervalos sao definidos como:

```python
OPEN_SLICE = slice(153, 165)        # indices 153-164 inclusive
MC_SLICE   = slice(1600, 1723)      # indices 1600-1722 inclusive
```
