# Guia de Deploy e Instalação — Olivia Platform

## 1) Backend (Edge + Fog + Cloud)

### Opção A: Docker Compose (recomendado para deploy)

Na raiz do projeto:

```bash
docker compose up --build -d
```

Serviços esperados:

- Edge: `http://<host>:8001`
- Fog: `http://<host>:8002`
- Cloud: `http://<host>:8080`

Portal web unificado:

- `http://<host>:8002/control/dashboard`

### Opção B: Execução manual (ambiente Python)

Em três terminais separados:

```bash
# Cloud
cd cloud_federated
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.inference.api:app --host 0.0.0.0 --port 8080
```

```bash
# Fog
cd fog_orchestrator
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8002
```

```bash
# Edge
cd edge_service
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

## 2) Banco de dados

O Fog usa SQLite local em:

- `fog_orchestrator/data/federated_local.db`

Tabelas principais:

- `users` (login/cadastro)
- `auth_sessions` (sessões)
- `captures_local` (coletas)
- `local_rounds` e `personal_rounds` (métricas de treino)

## 3) Dashboard Web (responsivo)

URL principal do painel:

- `http://<host>:8002/` (redireciona para `/control/dashboard`)

Dashboards detalhados:

- `http://<host>:8002/evaluation/dashboard`
- `http://<host>:8002/federated/dashboard`

## 4) Aplicativo Flutter (instalável)

### Android (APK)

```bash
cd mobile_app
flutter pub get
flutter build apk --release
```

APK gerado em:

- `mobile_app/build/app/outputs/flutter-apk/app-release.apk`

### Web (PWA básica)

```bash
cd mobile_app
flutter build web --release
```

Saída em:

- `mobile_app/build/web`

### iOS

```bash
cd mobile_app
flutter build ios --release
```

## 5) Configuração no app

No login/cadastro, informe a URL do Fog (ex.: `http://10.0.2.2:8002` no emulador Android).
No menu, configure URLs de Edge/Fog/Cloud conforme seu ambiente.

## 6) Endpoints de autenticação

- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me?token=<token>`
- `POST /auth/logout`

## 7) Observações de produção

- Coloque um proxy reverso (Nginx/Traefik) com HTTPS na frente de Edge/Fog/Cloud.
- Restrinja CORS e rotas administrativas por ambiente.
- Para escala e alta disponibilidade, migre SQLite para PostgreSQL.
- Configure backup periódico de banco e de modelos (`fog_orchestrator/models`).
