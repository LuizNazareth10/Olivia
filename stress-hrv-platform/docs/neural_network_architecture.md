# Arquitetura da Rede Neural (BiLSTM + Atenção)

Este documento descreve a arquitetura de inferência/treino utilizada no projeto para estimativa de probabilidade de estresse com base em métricas de HRV.

## 1) Resumo estrutural

- Entrada temporal: sequência de 30 passos com 4 features por passo.
- Backbone: BiLSTM bidirecional (2 camadas, hidden 64 por direção).
- Mecanismo de atenção temporal para ponderar quais instantes da janela têm maior contribuição.
- Cabeça de classificação com saída sigmoide para probabilidade em [0, 1].

## 2) Diagrama Mermaid

```mermaid
flowchart TD
    A[Input HRV sequence\nshape B x 30 x 4\nfeatures rmssd sdnn pnn50 mean_hr] --> B[BiLSTM\ninput_dim 4\nhidden_dim 64\nlayers 2\nbidirectional true]
    B --> C[Temporal output\nshape B x 30 x 128]
    C --> D[Attention MLP\nLinear 128 to 64\nTanh\nLinear 64 to 1]
    D --> E[Attention weights\nSoftmax over time axis]
    E --> F[Context vector\nweighted sum\nshape B x 128]
    F --> G[Classifier\nLinear 128 to 64\nReLU\nDropout 0.2\nLinear 64 to 1\nSigmoid]
    G --> H[Stress probability\nshape B\nrange 0 to 1]

    H --> I{Decision threshold}
    I -->|p >= 0.70| J[high_stress = 1]
    I -->|p < 0.70| K[high_stress = 0]
```

Arquivo editável do diagrama: `docs/diagrams/bilstm_attention_architecture.mmd`.

## 3) Dimensionalidade por camada

- Input: `B x 30 x 4`
- BiLSTM output: `B x 30 x 128`
- Attention scores: `B x 30 x 1`
- Attention weights: `B x 30 x 1`
- Context vector: `B x 128`
- Classifier output: `B x 1`
- Final output (squeeze): `B`

## 4) Justificativa técnica

- A componente BiLSTM modela dependências temporais da dinâmica HRV em janela curta.
- A atenção melhora interpretabilidade parcial ao explicitar quais instantes da janela tiveram maior peso na decisão.
- A cabeça sigmoide simplifica a integração operacional com limiar clínico/operacional (`0.70`).

## 5) Observações para apresentação

- A arquitetura atual trabalha com features agregadas de HRV por janela, não com waveform PPG bruto.
- Isso favorece robustez computacional e menor custo de inferência em borda/fog.
- Como extensão, pode-se adicionar camada de calibração de probabilidade por perfil etário/condicionamento.
