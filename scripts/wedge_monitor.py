"""While an e2e session is streaming, sample low-level state of the web app.

Samples every 2s for N seconds:
  - main-thread wchan of the web app (blocked sync call vs ep_poll idle)
  - TCP Recv-Q/Send-Q of the web app's browser-ws (:8090) and turn-ws (:8000) sockets
  - thread count of the web app
Usage: venv/bin/python -I scripts/wedge_monitor.py <web_pid> <seconds>
"""
import subprocess
import sys
import time

pid = sys.argv[1]
dur = float(sys.argv[2])
t0 = time.time()
while time.time() - t0 < dur:
    ts = time.strftime("%H:%M:%S")
    wchan = (
        subprocess.run(
            ["cat", f"/proc/{pid}/task/{pid}/wchan"], capture_output=True, text=True
        ).stdout.strip()
    )
    nthreads = len(
        subprocess.run(["ls", f"/proc/{pid}/task"], capture_output=True, text=True)
        .stdout.split()
    )
    ss = subprocess.run(
        ["ss", "-tnp"], capture_output=True, text=True
    ).stdout
    conns = []
    for line in ss.splitlines():
        if f"pid={pid}" not in line:
            continue
        # keep port and queues
        parts = line.split()
        if len(parts) >= 4:
            conns.append(f"{parts[3]} recvq={parts[0]} sendq={parts[1]}")
    print(
        f"{ts} wchan={wchan} threads={nthreads} | " + " ; ".join(conns),
        flush=True,
    )
    time.sleep(2)
