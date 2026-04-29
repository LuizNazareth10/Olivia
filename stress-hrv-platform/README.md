# Stress HRV Platform (Edge-Fog-Cloud + Federated Learning)

Projeto completo para captura de sinal PPG por câmera de celular, extração de HRV, predição de estresse e treinamento federado com **BiLSTM + Atenção**.

## Arquitetura

- **Edge (app + edge service):** captura de vídeo orientada, extração local de HRV e inferência rápida.
- **Fog:** orquestra decisões, agrega eventos e encaminha para o cloud.
- **Cloud:** mantém o modelo global e coordena rounds de aprendizado federado.

## Estrutura

- `mobile_app/`: App Flutter (Android/iOS).
- `edge_service/`: API FastAPI para processamento de HRV e inferência edge.
- `fog_orchestrator/`: API FastAPI de orquestração fog.
- `cloud_federated/`: Treino federado (Flower + PyTorch), export e API global.
- `docs/`: documentação de arquitetura.

## 1) Pré-requisitos

- Flutter SDK 3.24+
- Python 3.11+
- Docker Desktop (opcional)
- Android Studio (emulador/dispositivo)

## 2) Rodar backend local (sem Docker)

Em 3 terminais separados:

### Cloud API

```bash
cd cloud_federated
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn src.inference.api:app --host 0.0.0.0 --port 8080
```

### Fog

```bash
cd fog_orchestrator
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8002
```

### Edge

```bash
cd edge_service
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

## 3) Rodar app mobile

```bash
cd mobile_app
flutter create .
flutter pub get
flutter run
```

No app, configure a URL do Edge Service para o IP da sua máquina na rede local (ex.: `http://192.168.0.10:8001`).

## 4) Treinamento federado (simulação)

Em `cloud_federated`:

```bash
python src/federated/simulate.py --rounds 5 --clients 8 --local-epochs 2
```

Arquivos gerados:

- `models/global_model.pt`
- `models/global_model.onnx`
- `models/global_model.tflite` (placeholder para conversão posterior em pipeline mobile)

## 5) Fluxo funcional

1. Usuário inicia captura guiada (dedo sobre câmera/flash) por ~30s.
2. App extrai sinal bruto e calcula métricas HRV (RMSSD, SDNN, pNN50, LF/HF aproximado).
3. Features são inferidas localmente e também enviadas ao Edge/Fog/Cloud.
4. Se risco de estresse > limiar, app dispara notificação imediata.
5. Treinamento federado atualiza modelo global sem enviar dado bruto do usuário.

## 6) Coletas reais + federated rounds automáticos

- Cada chamada ao endpoint `edge /analyze` agora deve incluir `device_id`.
- O pipeline `edge -> fog -> cloud` persiste cada coleta no banco SQLite do cloud em:
	- `cloud_federated/data/federated_store.db`
- O cloud mantém:
	- histórico de coletas,
	- estado por dispositivo (`device_id -> last_model_version`),
	- histórico de rounds federados.
- Quando existem dados suficientes de múltiplos dispositivos, o cloud executa round federado automático e incrementa `model_version`.

Endpoint de acompanhamento:

- `GET http://localhost:8080/federated/status`
	- mostra `model_version`, `rounds`, `samples_total`, pendências e mapeamento por `device_id`.

## 7) Modo federado estrito (sem dado de usuário na cloud)

- No modo federado estrito:
	- dados por usuário/dispositivo ficam no `fog` (banco local),
	- cloud recebe apenas atualização de pesos do modelo + métricas agregadas de round.
- Status do coordenador cloud:
	- `GET http://localhost:8080/federated/status`
- Status local no fog:
	- `GET http://localhost:8002/federated/local-status`
- Dashboard operacional federado (com IDs de dispositivo locais):
	- `http://localhost:8002/federated/dashboard`

Treino por usuário (personalizado):

- Cada `device_id` tem modelo pessoal treinado automaticamente no fog a cada **50 coletas**.
- Antes de 50 coletas, a inferência usa modelo global no fog (com fallback heurístico, se necessário).
- Após 50+ coletas, a inferência passa a combinar modelo pessoal + global + probabilidade local.

Dashboard de avaliação por usuário (executa no fog com dados locais):

- `http://localhost:8002/evaluation/dashboard`

Endpoints de avaliação local:

- `GET http://localhost:8002/evaluation/summary`
- `GET http://localhost:8002/evaluation/series?device_id=<id>&horizon=1`

## Observações importantes (científicas)

- O pipeline fornecido é **base funcional de pesquisa**.
- Para uso clínico, valide com protocolo IRB/CEP, datasets maiores e calibração individual robusta.
- Extração PPG por câmera depende de iluminação, dispositivo e estabilidade do dedo.

## Documentação para apresentação acadêmica

- Arquitetura da rede neural (BiLSTM + Atenção):
	- `docs/neural_network_architecture.md`
	- Diagrama editável Mermaid: `docs/diagrams/bilstm_attention_architecture.mmd`
	- Diagrama em PNG: `docs/diagrams/bilstm_attention_architecture.png`
- Relatório técnico do experimento (global + pessoal):
	- `docs/training_analysis_report.md`
- Guia de deploy e instalação (backend + dashboard + app):
	- `docs/deployment_guide.md`
