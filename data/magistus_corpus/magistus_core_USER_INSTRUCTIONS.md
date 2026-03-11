# Magistus — User Instructions

## Prerequisites

- **Python 3.12 or newer** — [Download](https://www.python.org/downloads/)
  - Ensure `python` and `pip` are on your PATH
- **Node.js 18 or newer** — [Download](https://nodejs.org/)
  - Comes with `npm`
- **OpenAI API key** — [Get one here](https://platform.openai.com/api-keys)
  - Needs GPT-4 access

## Install

### Automatic (recommended)

Run the boot script from the project root in PowerShell:

```powershell
.\boot.ps1
```

This handles everything: venv, pip install, npm install, frontend build, .env creation, and server start.

### Manual

```bash
# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # Linux/Mac

# Install Python dependencies
pip install -r requirements.txt

# Build the React frontend
cd magistus-ui
npm install
npm run build
cd ..
```

## Credential/API Setup

Magistus requires an OpenAI API key for GPT-4 and embedding calls.

1. Get your key from https://platform.openai.com/api-keys
2. Create a `.env` file in the project root (see next section)
3. The key is never logged or transmitted anywhere except OpenAI's API

## Configure `.env`

Create a file named `.env` in the project root:

```env
OPENAI_API_KEY=sk-your-key-here
```

Optional variables:

```env
# Change the embedding model (default: text-embedding-3-small)
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Change the chat model (default: gpt-4)
OPENAI_CHAT_MODEL=gpt-4
```

> **Important:** Never commit `.env` to source control. It is listed in `.gitignore`.

## Run

Start the backend server:

```bash
python -m uvicorn launch_v1:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000** in your browser.

For development with hot-reload:

```bash
# Terminal 1 — backend
python -m uvicorn launch_v1:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 — frontend dev server (auto-proxies API calls)
cd magistus-ui
npm run dev
```

## Stop

Press `Ctrl+C` in the terminal running Uvicorn.

If you used `boot.ps1`, press `Ctrl+C` in the PowerShell window.

## Key Settings Explained

All settings live in `config.yaml`:

| Setting | Default | Description |
|---------|---------|-------------|
| `agents_enabled` | (list) | Which cognitive agents participate in reasoning |
| `allow_self_eval` | `true` | Allow agents to self-assess and suggest improvements |
| `voice_output` | `true` | Enable text-to-speech in responses |
| `debug_mode` | `true` | Verbose logging to console |
| `web_search_enabled` | `true` | Allow web search agent to query Google |
| `grounding_layer_enabled` | `true` | Enable factual grounding checks |
| `consent_protocols_enabled` | `true` | Require user consent for critical actions |
| `transparency_layer_enabled` | `true` | Show agent reasoning transparency |
| `min_confidence_threshold` | `0.85` | Minimum confidence for agent responses |
| `limiter_enabled` | `false` | Enable the synthetic limiter (blocks risky inputs) |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` on startup | Activate your venv: `.venv\Scripts\activate`, then `pip install -r requirements.txt` |
| `OPENAI_API_KEY not set` | Create a `.env` file with your key (see Configure section) |
| Frontend shows "UI not found" | Run `cd magistus-ui && npm run build` to generate the dist/ folder |
| Port 8000 already in use | Kill the existing process or use `--port 8001` |
| `npm install` fails | Ensure Node.js 18+ is installed: `node --version` |
| Agent returns empty response | Check that the agent is listed in `config.yaml` `agents_enabled` |
| Voice output not working | Ensure `voice_output: true` in config.yaml and speakers are connected |
| Tests fail with import errors | Run from project root with venv activated: `.venv\Scripts\python.exe -m pytest tests/` |
