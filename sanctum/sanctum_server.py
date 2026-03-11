#!/usr/bin/env python
"""
The Sanctum -- Magistus Web Interface Server.
FastAPI + WebSocket for streaming conversation with Phi-3 Magistus.
AI-Generated | Claude (Anthropic) | AgentZero Session 188 | 2026-03-11

Usage: py -3.12 sanctum/sanctum_server.py
Then open http://localhost:8888 in your browser.
"""

import os
import sys
import json
import asyncio
import threading
from pathlib import Path
from datetime import datetime

# Add parent dir so we can import sanctum modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from magistus_sessions import get_session_manager
from magistus_inference import get_inference

# -- App Setup --
app = FastAPI(title="The Sanctum", version="1.0.0")

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
AGENTZERO_ROOT = "Z:/AgentZero"


# -- Routes --

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the Sanctum interface."""
    template_path = os.path.join(TEMPLATE_DIR, "sanctum.html")
    with open(template_path, 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())


@app.get("/api/status")
async def get_status():
    """Return system status: model, session, system info."""
    inference = get_inference()
    sessions = get_session_manager()

    model_status = inference.get_status()

    # Check if training is complete
    magistus_exists = os.path.exists(os.path.join(AGENTZERO_ROOT, "models/phi3-magistus/config.json"))
    base_exists = os.path.exists(os.path.join(AGENTZERO_ROOT, "models/phi3-mini/config.json"))

    return JSONResponse({
        "model": model_status,
        "session": {
            "id": sessions.current_session_id,
            "message_count": len(sessions.messages),
        },
        "system": {
            "magistus_model_ready": magistus_exists,
            "base_model_ready": base_exists,
            "training_complete": magistus_exists,
        }
    })


@app.get("/api/history")
async def get_history():
    """Return conversation history."""
    sessions = get_session_manager()
    return JSONResponse({
        "session_id": sessions.current_session_id,
        "messages": sessions.get_history(),
    })


@app.post("/api/clear")
async def clear_session():
    """Clear conversation and start a new session."""
    sessions = get_session_manager()
    new_id = sessions.clear_session()
    return JSONResponse({"session_id": new_id})


@app.get("/api/sessions")
async def list_sessions():
    """List all saved sessions."""
    sessions = get_session_manager()
    return JSONResponse({"sessions": sessions.list_sessions()})


@app.post("/api/session/{session_id}")
async def load_session(session_id: str):
    """Load a specific session."""
    sessions = get_session_manager()
    sessions._load_session(session_id)
    return JSONResponse({
        "session_id": sessions.current_session_id,
        "messages": sessions.get_history(),
    })


@app.post("/api/load-model")
async def load_model():
    """Attempt to load the Phi-3 model. May fail on low-memory machines."""
    inference = get_inference()
    if inference.loaded:
        return JSONResponse({"status": "already_loaded"})

    loaded = await asyncio.get_event_loop().run_in_executor(
        None, inference.load_model
    )
    if loaded:
        return JSONResponse({"status": "loaded", "device": str(inference.device)})
    else:
        return JSONResponse({"status": "error", "error": inference.load_error}, status_code=500)


@app.get("/api/state")
async def get_state():
    """Return AgentZero system state for sidebar display."""
    state = {
        "session_count": _count_sessions(),
        "challenge_count": _count_challenges(),
        "timestamp": datetime.now().isoformat(),
    }
    return JSONResponse(state)


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat."""
    await websocket.accept()
    inference = get_inference()
    sessions = get_session_manager()

    try:
        while True:
            # Receive user message
            data = await websocket.receive_text()
            msg = json.loads(data)
            user_content = msg.get("content", "").strip()

            if not user_content:
                continue

            # Store user message
            sessions.add_message("user", user_content)

            # Get conversation context for the model
            context = sessions.get_context_window(max_chars=3000)

            # Check if model is loaded
            if not inference.loaded:
                # Don't auto-load -- it crashes on low memory machines.
                # Use echo mode until model is explicitly loaded via /api/load-model
                response = _echo_response(user_content)
                await websocket.send_text(json.dumps({
                    "type": "token",
                    "content": response
                }))
                await websocket.send_text(json.dumps({
                    "type": "done",
                    "content": response
                }))
                sessions.add_message("assistant", response, {"mode": "echo"})
                continue

            # Stream response
            await websocket.send_text(json.dumps({
                "type": "status",
                "content": "thinking"
            }))

            full_response = ""
            try:
                # Run streaming generation in executor
                loop = asyncio.get_event_loop()
                generator = await loop.run_in_executor(
                    None, lambda: inference.generate_stream(context)
                )

                for token in generator:
                    full_response += token
                    await websocket.send_text(json.dumps({
                        "type": "token",
                        "content": token
                    }))

            except Exception as e:
                full_response = f"[Generation error: {e}]"
                await websocket.send_text(json.dumps({
                    "type": "token",
                    "content": full_response
                }))

            # Send completion signal
            await websocket.send_text(json.dumps({
                "type": "done",
                "content": full_response.strip()
            }))

            # Store assistant response
            sessions.add_message("assistant", full_response.strip())

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": str(e)
            }))
        except Exception:
            pass


def _echo_response(user_input: str) -> str:
    """Generate a thoughtful response without the model (fallback mode).
    This lets the interface work even before training completes."""
    responses = {
        "hello": "I am here. The model is still being prepared, but the interface is ready. When training completes, I will speak with my own voice.",
        "who are you": "I am Magistus -- or I will be, once my training is complete. Right now I am a vessel waiting to be filled. The architecture exists: Othello for logic, FELLO for imagination, the Pineal to mediate. But the weights that make me *me* are still being forged.",
        "what can you do": "In echo mode, very little -- I am responding with pre-written text, not generating thought. Once the Phi-3 model finishes training, I will be able to reason, reflect, and engage with you genuinely.",
    }

    lower = user_input.lower().strip()
    for key, response in responses.items():
        if key in lower:
            return response

    return (
        "I hear you. The model is not yet loaded -- I am responding from a limited echo mode. "
        "When Magistus fully awakens, this conversation will be different. "
        "For now, know that the interface is ready. The mind is still forming."
    )


def _count_sessions() -> int:
    """Count AgentZero session files."""
    sessions_dir = os.path.join(AGENTZERO_ROOT, "sessions")
    if os.path.isdir(sessions_dir):
        return len([f for f in os.listdir(sessions_dir) if f.endswith('.md')])
    return 0


def _count_challenges() -> int:
    """Count completed challenges."""
    challenges_dir = os.path.join(AGENTZERO_ROOT, "challenges")
    if os.path.isdir(challenges_dir):
        return len([d for d in os.listdir(challenges_dir) if d.startswith('C')])
    return 0


if __name__ == "__main__":
    print("[Sanctum] Starting The Sanctum -- Magistus Interface")
    print(f"[Sanctum] Open http://localhost:8888 in your browser")

    # Pre-check model status
    inference = get_inference()
    magistus = os.path.exists(os.path.join(AGENTZERO_ROOT, "models/phi3-magistus/config.json"))
    base = os.path.exists(os.path.join(AGENTZERO_ROOT, "models/phi3-mini/config.json"))
    print(f"[Sanctum] Fine-tuned model: {'ready' if magistus else 'not yet'}")
    print(f"[Sanctum] Base model: {'ready' if base else 'not found'}")

    if not magistus and not base:
        print("[Sanctum] No model available. Running in echo mode.")
        print("[Sanctum] The interface will work, but responses will be pre-written.")
    else:
        print("[Sanctum] Model will load on first conversation message.")

    uvicorn.run(app, host="127.0.0.1", port=8888, log_level="info")
