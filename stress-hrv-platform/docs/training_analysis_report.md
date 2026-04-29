# Relatório técnico de análise do treinamento (global + pessoal)

## Escopo

Este relatório consolida o experimento de carga sintética solicitado para avaliação acadêmica com foco em HRV:

- 5 perfis fisiológicos distintos.
- 250 coletas por perfil.
- Treino de modelos pessoais por perfil.
- Treino do modelo global via rounds federados locais no fog.

## 1) Protocolo de geração de dados

### Perfis simulados

1. Perfil 1 — Atleta de endurance.
2. Perfil 2 — Adulto ativo recreacional.
3. Perfil 3 — Sedentário com sobrepeso.
4. Perfil 4 — Indivíduo sob estresse crônico.
5. Perfil 5 — Idoso saudável ativo.

### Variáveis usadas no treino

A geração e o treino usam métricas derivadas de HRV:

- rmssd
- sdnn
- pnn50
- mean_hr

### Volume gerado

- Total novo inserido no experimento: 1250 amostras (5 x 250).
- Cada perfil foi associado a um par estável user_id/device_id para permitir rastreio de contribuição por usuário.

## 2) Pipeline de treinamento

### 2.1 Treino de modelos pessoais

- Disparo de treino pessoal por dispositivo após atingir volume mínimo.
- Resultado do experimento: 5 modelos pessoais treinados (um por perfil), cada um com 250 amostras.

### 2.2 Treino do modelo global

- Estratégia federada local (fog) com agregação de clientes elegíveis.
- Após inserção, foram executados 2 rounds globais adicionais no experimento, com 5 clientes participantes.

## 3) Métricas obtidas

## 3.1 Modelo global (último round)

- round_version: 4
- clients: 5
- samples: 610
- MAE: 0.429726
- RMSE: 0.429824
- sMAPE: 1.528290
- MASE: 22441268.785246

## 3.2 Modelos pessoais (último por perfil)

- device_endurance: MAE 0.423041, RMSE 0.423089, amostras 250
- device_active: MAE 0.387493, RMSE 0.387549, amostras 250
- device_sedentary: MAE 0.517688, RMSE 0.518382, amostras 250
- device_chronic_stress: MAE 0.527101, RMSE 0.527119, amostras 250
- device_senior: MAE 0.385080, RMSE 0.385094, amostras 250

## 4) Interpretação crítica (nível acadêmico)

### 4.1 O que os resultados sugerem

- O experimento validou a capacidade do pipeline em escalar para múltiplos perfis e treinar global + pessoais no mesmo ciclo.
- Os modelos pessoais mostram diferenciação coerente entre perfis fisiológicos mais estáveis e perfis com maior carga de estresse.
- O modelo global converge após rounds adicionais, mantendo métricas estáveis entre rounds recentes.

### 4.2 Limitações metodológicas atuais

- O rótulo `high_stress` é derivado por limiar da própria probabilidade (`>= 0.70`), caracterizando cenário de pseudo-rotulagem.
- Métricas em dados sintéticos podem superestimar generalização para sinais reais em campo.
- O valor de MASE pode ficar inflado em cenários com baixa variação temporal na série-alvo (denominador pequeno).

### 4.3 Recomendações para banca/professores

- Destacar que o trabalho está em nível de protótipo de pesquisa aplicada, com arquitetura funcional e observabilidade de treino.
- Para validação científica forte, adicionar ground truth externo (questionário validado, protocolo experimental, eventos anotados).
- Incluir calibração por subgrupo (idade/condicionamento) e validação temporal prospectiva.
- Reportar também métricas de calibração (Brier Score, ECE) além de erro pontual.

## 5) Onde consultar no sistema

- Status local do fog: `GET /federated/local-status`
- Métricas de treino global + pessoal: `GET /federated/training-metrics?limit=10`
- Geração dos perfis (endpoint do experimento): `POST /federated/simulate-profiles`

## 6) Reprodutibilidade

Parâmetros de execução usados no experimento:

- samples_per_profile: 250
- sampling_hz: 30
- seed: 20260225
- run_global_training: true

Com isso, o cenário fica reproduzível no mesmo ambiente de execução e banco local.
