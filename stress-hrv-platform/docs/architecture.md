# Arquitetura Edge-Fog-Cloud para HRV + Estresse

## Edge (dispositivo + API edge)

- Captura de vídeo PPG via câmera e processamento instantâneo de sinal.
- Cálculo de métricas HRV de curta janela (RMSSD, SDNN, pNN50 e HR médio).
- Inferência local para baixa latência e resiliência offline.

## Fog (orquestração intermediária)

- Recebe escore local do edge.
- Consulta o modelo global no cloud.
- Faz fusão de probabilidade (`0.6 * cloud + 0.4 * edge`) e retorna risco consolidado.

## Cloud (modelo global federado)

- Mantém o modelo **BiLSTM + Atenção**.
- Executa rounds federados simulados por FedAvg com perfis de usuário heterogêneos.
- Distribui pesos globais mais recentes para inferência.

## Privacidade e FL

- Dados brutos de sinal não precisam sair do dispositivo no cenário ideal.
- Treinamento federado troca apenas parâmetros/gradientes agregados.
- Novos usuários podem iniciar com o modelo global sem histórico pessoal.

## Notificações de estresse

- Limiar padrão de alerta: `P(estresse) >= 0.70`.
- Disparo local por notificação no celular.
- Recomendação UX: alerta explicativo e ação breve (respiração/pausa/hidratação).
