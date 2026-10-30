"""Thread-safe WebSocket event push (extracted from app.py 2026-10-30).

emit_to_room() is called from worker threads (speech pipeline, pvad gates,
agent-runtime delivery, VAD callbacks) as well as the asyncio loop, so every
send is marshalled onto the app's main event loop via
asyncio.run_coroutine_threadsafe. The two things this module must not own —
the session registry and the loop itself — are wired in by app.py (a direct
import of app from here would be circular).
"""
import asyncio
import json
import logging

from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)

# The app's main event loop — set by app.py's startup event.
main_loop = None

# Session lookup (client_id -> ChatSession) — wired by app.py right after
# the SessionManager is created.
_get_session = lambda client_id: None


def set_session_lookup(fn):
    """Wire the session lookup used by emit_to_room."""
    global _get_session
    _get_session = fn


def emit_to_room(client_id, event, data):
    """Helper function to safely emit WebSocket messages to a specific client."""
    session = _get_session(client_id)
    if not session or not main_loop:
        return

    ws = session.websocket

    if ws.client_state != WebSocketState.CONNECTED:
        return

    # Helper wrapper to run async send in the main loop
    async def _send():
        try:
            if event == "audio_chunk":
                # Audio data: send raw bytes
                await ws.send_bytes(data)
            else:
                # Text/JSON data
                message = json.dumps({"event": event, "data": data})
                await ws.send_text(message)
        except Exception as e:
            logger.error(f"Failed to send to {client_id}: {e}")

    asyncio.run_coroutine_threadsafe(_send(), main_loop)
