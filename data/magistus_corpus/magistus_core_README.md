# Magistus

An AI reasoning companion built on a multi-agent cognitive architecture. Magistus orchestrates specialized agents — each modeled after brain regions — to produce thoughtful, transparent, and ethically grounded responses.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  React Frontend  (magistus-ui/)                     │
│  ┌───────┐ ┌──────────┐ ┌────────┐ ┌────────────┐  │
│  │Header │ │  Chat    │ │ Input  │ │ Feedback   │  │
│  │       │ │  + Agent │ │ Form   │ │ Modal      │  │
│  │       │ │  Panels  │ │        │ │            │  │
│  └───────┘ └──────────┘ └────────┘ └────────────┘  │
└────────────────────┬────────────────────────────────┘
                     │  HTTP / Streaming
┌────────────────────▼────────────────────────────────┐
│  FastAPI Backend  (launch_v1.py)                    │
│  /conversation  /chat  /think  /reflect  /toggle    │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  Central Hub  (central_hub.py)                      │
│  Orchestrates agents, fuses responses               │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  Agent Layer  (agents/)                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │Temporal  │ │Prefrontal│ │Default   │  ...        │
│  │Lobe      │ │Cortex    │ │Mode Net  │             │
│  └──────────┘ └──────────┘ └──────────┘             │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  Support Layers                                     │
│  Memory · Ethics · Transparency · Policy · Limiter  │
└─────────────────────────────────────────────────────┘
```

## Prerequisites

- **Python 3.12+**
- **Node.js 18+** (with npm)
- **OpenAI API key** (GPT-4 access)

## Quick Start

### One-command setup (Windows PowerShell)

```powershell
.\boot.ps1
```

This creates the virtual environment, installs dependencies, builds the frontend, and starts the server.

### Manual setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Build the React frontend
cd magistus-ui
npm install
npm run build
cd ..

# 4. Set up environment variables
#    Create a .env file at project root:
echo OPENAI_API_KEY=sk-your-key-here > .env

# 5. Start the server
python -m uvicorn launch_v1:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** in your browser.

## Folder Structure

```
magistus/
├── launch_v1.py              # FastAPI entry point
├── central_hub.py            # Agent orchestrator
├── config.yaml               # Agent/feature toggle config
├── context_types.py          # Shared data models (AgentThought, ContextBundle)
├── config_loader.py          # YAML config loader
├── requirements.txt          # Python dependencies
├── boot.ps1                  # One-click setup script
├── agents/                   # Cognitive agents
│   ├── temporal_lobe.py      # Memory-contextualized reasoning
│   ├── prefrontal_cortex.py  # Executive reasoning
│   ├── default_mode_network.py  # Creative/associative thinking
│   ├── anterior_cingulate.py # Conflict monitoring
│   ├── reflective_self_monitor.py  # Self-assessment
│   ├── web_search_agent.py   # External search
│   └── ...
├── memory/                   # Memory subsystem
├── meta_learning/            # Meta-learning & reflection
├── ethical/                  # Ethics engine
├── cognitive_extensions/     # Confidence, emotion, self-assessment
├── profile/                  # User & system profiles
├── magistus-ui/              # React frontend (Vite + TypeScript)
│   ├── src/
│   │   ├── App.tsx           # Root component
│   │   ├── components/       # Chat, Header, InputForm, etc.
│   │   └── services/api.ts   # Backend API client
│   └── dist/                 # Built frontend (served by backend)
├── tests/                    # pytest test suite
├── Forge/                    # Build governance framework
└── logs/                     # Runtime logs
```

## Development Workflow

### Running tests

```bash
# Backend (pytest)
.venv\Scripts\python.exe -m pytest tests/ -q

# Frontend (Vitest)
cd magistus-ui && npm test
```

### Development server (with hot reload)

```bash
# Terminal 1: Backend
python -m uvicorn launch_v1:app --reload --host 127.0.0.1 --port 8000

# Terminal 2: Frontend dev server (proxies to backend)
cd magistus-ui && npm run dev
```

### Key configuration

Edit `config.yaml` to enable/disable agents and features:

```yaml
agents_enabled:
  - temporal_lobe
  - prefrontal_cortex
  - default_mode_network
  # ...

allow_self_eval: true    # Toggles autonomous self-evaluation
voice_output: true       # Text-to-speech output
debug_mode: true         # Verbose logging
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve React UI |
| POST | `/conversation` | Main conversation (streaming or reasoning) |
| POST | `/chat` | Legacy chat endpoint |
| POST | `/think` | Direct agent reasoning |
| POST | `/reflect` | Trigger meta-learning reflection |
| POST | `/toggle_self_eval` | Toggle self-evaluation mode |
| GET | `/fetch_reasoning/{id}` | Retrieve cached reasoning results |

## License

Private project — all rights reserved.
