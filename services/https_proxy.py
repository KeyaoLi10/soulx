"""TLS termination proxy for company-intranet access to the SoulX web app.

Why: getUserMedia (microphone) and SharedArrayBuffer (TTS ring buffer) are
Secure-Context APIs — they only exist on https:// or http://localhost. The
web app itself is plain HTTP on :8090, so intranet users hitting
http://<lan-ip>:8090 get "Audio Engine initialization failed" and no mic.
The tailscale path works because it terminates TLS. This proxy does the same
for the LAN: https://<lan-ip>:8444 -> 127.0.0.1:8090 (plain byte pipe, so
WebSocket upgrades pass through untouched).

The certificate is self-signed (certs/intranet-cert.pem, SANs = all local
IPs): browsers show a one-time warning that users click through. If a proper
internal CA is available later, drop its cert/key in and restart.

Usage:
  venv/bin/python services/https_proxy.py [--port 8444] [--target 127.0.0.1:8090]
"""
import argparse
import asyncio
import ssl
from pathlib import Path

CERT = Path(__file__).parent / "certs" / "intranet-cert.pem"
KEY = Path(__file__).parent / "certs" / "intranet-key.pem"


async def pipe(reader, writer, direction, peer):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def handle(client_reader, client_writer, target_host, target_port):
    peer = client_writer.get_extra_info("peername")
    try:
        upstream_reader, upstream_writer = await asyncio.wait_for(
            asyncio.open_connection(target_host, target_port), timeout=5
        )
    except Exception as exc:
        print(f"[https-proxy] upstream connect failed for {peer}: {exc}", flush=True)
        client_writer.close()
        return
    await asyncio.gather(
        pipe(client_reader, upstream_writer, "c->s", peer),
        pipe(upstream_reader, client_writer, "s->c", peer),
    )


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8444)
    ap.add_argument("--target", default="127.0.0.1:8090")
    args = ap.parse_args()
    host, _, port = args.target.partition(":")
    target_port = int(port or 8090)

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(CERT, KEY)

    server = await asyncio.start_server(
        lambda r, w: handle(r, w, host, target_port),
        host="0.0.0.0",
        port=args.port,
        ssl=ctx,
    )
    print(
        f"[https-proxy] listening on 0.0.0.0:{args.port} (TLS) "
        f"-> {host}:{target_port}",
        flush=True,
    )
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
