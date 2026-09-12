#!/usr/bin/env python3
# Auto-generated for TunnelMate (https://163.61.236.112/llms.txt)
# Kaggle VM runs an SSH server + tunnelmate-agent. The agent dials OUT to the
# broker, which relays traffic from a public port back to the SSH server.
# Multiple concurrent SSH connections supported.
#
# After this script starts, connect from your machine:
#   ssh -p <PUBLIC_PORT> notebook@163.61.236.112
#   password: <SSH_PASSWORD>
#
# Stop the tunnel by creating the sentinel file: touch /tmp/shutdown_notebook

import asyncio, importlib, json, os, pty, shutil, struct, subprocess, sys
import termios, time, traceback, urllib.request
from pathlib import Path


def ensure_package(m, pkg=None):
    try:
        return importlib.import_module(m)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg or m])
        return importlib.import_module(m)


# ── Runtime deps (installed on first use) ──
ensure_package("aiohttp")
asyncssh = ensure_package("asyncssh")

# ── Configuration (edit me) ──
SSH_USER = "notebook"
SSH_PASSWORD = "notebook123"          # SSH password
SSH_HOST = "127.0.0.1"
SSH_PORT = 2222

TUNNELMATE_BROKER = "http://163.61.236.112"
BROKER_HOST = "163.61.236.112"
BROKER_CONTROL_PORT = 7000
SCOPE = "open"                              # "open" = anyone can reach it (SSH itself is password-protected)
PROTOCOL = "tcp"

WORK_DIR = Path("/kaggle/working/.tunnelmate")
if not WORK_DIR.exists():
    WORK_DIR = Path.home() / ".tunnelmate"
WORK_DIR.mkdir(parents=True, exist_ok=True)

AGENT_BIN = WORK_DIR / "tunnelmate-agent"
BROKER_CERT = WORK_DIR / "broker.crt"
AGENT_CONF = WORK_DIR / "agent.conf"
TUNNEL_STATE = WORK_DIR / "tunnel.json"     # saved secrets, reused on restart
TUNNEL_VERSION = "0.1.0"
SHUTDOWN_PATH = Path("/tmp/shutdown_notebook")
SHUTDOWN_POLL_SECONDS = 2


def _json_body(): return {"Content-Type": "application/json", "User-Agent": "kaggle-tunnelmate/0.1.0"}


# ── SSH server (native multi-connection, same as before) ──

class KaggleSSHServer(asyncssh.SSHServer):
    def connection_made(self, conn):
        peer = conn.get_extra_info("peername", ("?", 0))
        print(f"[ssh] connection from {peer[0]}:{peer[1]}")

    def begin_auth(self, _username):
        return True

    def password_auth_supported(self):
        return True

    def validate_password(self, username, password):
        ok = username == SSH_USER and password == SSH_PASSWORD
        print(f"[ssh] auth: user={username} -> {'OK' if ok else 'REJECTED'}")
        return ok

    def connection_lost(self, exc):
        print(f"[ssh] client disconnected: {exc or 'clean'}")


async def handle_ssh_client(process):
    env = os.environ.copy()
    if getattr(process, "term_type", None):
        env["TERM"] = process.term_type
    ts = getattr(process, "term_size", None)
    shell = shutil.which("bash") or env.get("SHELL") or "/bin/sh"

    def session_argv(command=None):
        """Start every SSH session with the user's Bash configuration loaded."""
        if os.path.basename(shell) == "bash":
            argv = [shell, "--noprofile", "--rcfile", str(Path.home() / ".bashrc"), "-i"]
        else:
            argv = [shell, "-i"]
        if command:
            argv.extend(["-c", command])
        return argv

    def apply_pty_size(fd, size=None):
        s = size or ts or (80, 24)
        try:
            w, h = int(s[0]), int(s[1])
            import fcntl
            fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", h, w, 0, 0))
        except Exception:
            pass

    def setup_child_pty(slave_fd):
        try:
            os.setsid()
        except Exception:
            pass
        try:
            import fcntl
            fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
        except Exception:
            pass

    if getattr(process, "term_type", None):
        master_fd, slave_fd = pty.openpty()
        apply_pty_size(slave_fd)
        child = subprocess.Popen(
            session_argv(process.command),
            stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
            env=env, close_fds=True,
            preexec_fn=lambda: setup_child_pty(slave_fd),
        )
        os.close(slave_fd)

        async def pump_ssh_to_pty():
            try:
                while True:
                    try:
                        chunk = await process.stdin.read(1024)
                    except asyncssh.TerminalSizeChanged as exc:
                        apply_pty_size(master_fd, exc.term_size)
                        continue
                    if not chunk:
                        break
                    if isinstance(chunk, str):
                        chunk = chunk.encode("utf-8", errors="replace")
                    await asyncio.to_thread(os.write, master_fd, chunk)
            except Exception:
                pass

        async def pump_pty_to_ssh():
            try:
                while True:
                    chunk = await asyncio.to_thread(os.read, master_fd, 65536)
                    if not chunk:
                        break
                    process.stdout.write(chunk)
                    await process.stdout.drain()
            except Exception:
                pass

        t1 = asyncio.create_task(pump_ssh_to_pty())
        t2 = asyncio.create_task(pump_pty_to_ssh())
        try:
            await asyncio.wait([t1, t2], return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in (t1, t2):
                if not t.done():
                    t.cancel()
            if child.poll() is None:
                child.terminate()
                try:
                    await asyncio.to_thread(child.wait)
                except Exception:
                    child.kill()
            os.close(master_fd)
        rc = child.returncode if child.returncode is not None else await asyncio.to_thread(child.wait)
        process.exit(rc)
        return

    child = subprocess.Popen(
        session_argv(process.command),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env,
    )
    await process.redirect(stdin=child.stdin, stdout=child.stdout, stderr=child.stderr)
    process.exit(await asyncio.to_thread(child.wait))


async def start_ssh():
    key_dir = WORK_DIR
    key_path = key_dir / "ssh_host_key"
    if not key_path.exists():
        asyncssh.generate_private_key("ssh-rsa").write_private_key(str(key_path))
    server = await asyncssh.create_server(
        KaggleSSHServer, SSH_HOST, SSH_PORT,
        server_host_keys=[str(key_path)],
        process_factory=handle_ssh_client, encoding=None,
    )
    print(f"[ssh] SSH on {SSH_HOST}:{SSH_PORT}  user={SSH_USER}  pass={SSH_PASSWORD}")
    return server


# ── TunnelMate agent setup ──

def download_agent():
    import platform as _platform
    machine = _platform.machine().lower()
    arch = "aarch64" if any(t in machine for t in ("aarch64", "arm64")) else "x86_64"
    url = f"{TUNNELMATE_BROKER}/v1/download/tunnelmate-{TUNNEL_VERSION}-linux-{arch}.tar.gz"
    print(f"[tm] Downloading agent ({arch}) from {url}")
    tar = WORK_DIR / "tunnelmate.tar.gz"
    urllib.request.urlretrieve(url, tar)
    subprocess.check_call(["tar", "xzf", str(tar), "-C", str(WORK_DIR)])
    bin_path = WORK_DIR / f"tunnelmate-{TUNNEL_VERSION}-linux-{arch}" / "tunnelmate-agent"
    bin_path.chmod(bin_path.stat().st_mode | 0o755)
    tar.unlink(missing_ok=True)
    print(f"[tm] Agent binary: {bin_path}")
    return bin_path


def fetch_broker_cert():
    try:
        urllib.request.urlretrieve(f"{TUNNELMATE_BROKER}/v1/broker-certificate", BROKER_CERT)
        print(f"[tm] Broker cert saved to {BROKER_CERT}")
        return str(BROKER_CERT)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print("[tm] Broker uses a publicly trusted cert; no ca_path needed")
            return None
        raise


def create_tunnel():
    if TUNNEL_STATE.exists():
        try:
            state = json.loads(TUNNEL_STATE.read_text())
            if state.get("scope") == SCOPE:
                print(f"[tm] Reusing saved tunnel {state['tunnel_id']}")
                return state
        except Exception:
            pass
    req = urllib.request.Request(
        f"{TUNNELMATE_BROKER}/v1/tunnels",
        data=json.dumps({"scope": SCOPE, "protocol": PROTOCOL}).encode(),
        headers=_json_body(), method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        state = json.loads(r.read())
    TUNNEL_STATE.write_text(json.dumps(state, indent=2))
    TUNNEL_STATE.chmod(0o600)
    print(f"[tm] Tunnel created: {state['tunnel_id']}")
    print(f"[tm]   public_port={state.get('public_port')}  peer_address={state.get('peer_address')}")
    return state


def write_agent_conf(state, ca_path):
    lines = [
        f"agent.tunnel_id = {state['tunnel_id']}",
        f"agent.agent_secret = {state['agent_secret']}",
        f"agent.local_host = {SSH_HOST}",
        f"agent.local_port = {SSH_PORT}",
        f"agent.broker_host = {BROKER_HOST}",
        f"agent.broker_port = {BROKER_CONTROL_PORT}",
        f"agent.protocol = {PROTOCOL}",
        f"agent.verify_ca = true",
    ]
    if ca_path:
        lines.append(f"agent.ca_path = {ca_path}")
    lines.append("agent.log_level = info")
    AGENT_CONF.write_text("\n".join(lines) + "\n")
    os.chmod(AGENT_CONF, 0o600)


def tunnel_online(state):
    req = urllib.request.Request(
        f"{TUNNELMATE_BROKER}/v1/tunnels/{state['tunnel_id']}",
        headers={"X-Tunnel-Management-Secret": state["management_secret"]},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        info = json.loads(r.read())
    return bool(info.get("online"))


def renew_tunnel(state):
    req = urllib.request.Request(
        f"{TUNNELMATE_BROKER}/v1/tunnels/{state['tunnel_id']}/renew",
        data=b"", method="POST",
        headers={"X-Tunnel-Management-Secret": state["management_secret"]},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def delete_tunnel(state):
    try:
        req = urllib.request.Request(
            f"{TUNNELMATE_BROKER}/v1/tunnels/{state['tunnel_id']}",
            method="DELETE",
            headers={"X-Tunnel-Management-Secret": state["management_secret"]},
        )
        urllib.request.urlopen(req, timeout=15)
        TUNNEL_STATE.unlink(missing_ok=True)
        print("[tm] Tunnel deleted")
    except Exception as exc:
        print(f"[tm] delete failed (lease will expire anyway): {exc}")


async def start_agent(bin_path, state, ca_path):
    if not AGENT_CONF.exists():
        write_agent_conf(state, ca_path)
    proc = await asyncio.create_subprocess_exec(
        str(bin_path), "-c", str(AGENT_CONF),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )

    async def drain():
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            print(f"  {line.decode('utf-8', errors='replace')}", end="")

    asyncio.create_task(drain())
    return proc


async def wait_online(state, timeout=60, shutdown_event=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if shutdown_event is not None and shutdown_event.is_set():
            return False
        try:
            if await asyncio.to_thread(tunnel_online, state):
                return True
        except Exception as exc:
            print(f"[tm] status check error: {exc}")
        if shutdown_event is None:
            await asyncio.sleep(3)
        else:
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=3)
            except asyncio.TimeoutError:
                pass
    return False


async def renew_loop(state):
    while True:
        await asyncio.sleep(12 * 3600)
        try:
            info = await asyncio.to_thread(renew_tunnel, state)
            print(f"[tm] Lease renewed, expires_at={info.get('expires_at')}")
        except Exception as exc:
            print(f"[tm] Renew failed: {exc}")


async def shutdown_watcher(shutdown_event):
    """Set shutdown_event when the shutdown sentinel file appears."""
    while not shutdown_event.is_set():
        if SHUTDOWN_PATH.exists():
            print(f"[watcher] Found {SHUTDOWN_PATH}; requesting shutdown ...")
            shutdown_event.set()
            return
        await asyncio.sleep(SHUTDOWN_POLL_SECONDS)


# ── Main ──

async def main():
    print("=" * 60)
    print("  Kaggle SSH via TunnelMate")
    print(f"  Broker: {TUNNELMATE_BROKER}  scope={SCOPE} protocol={PROTOCOL}")
    print("=" * 60)

    shutdown_event = asyncio.Event()
    watcher_task = asyncio.create_task(shutdown_watcher(shutdown_event))
    renew_task = None
    ssh_server = None
    proc = None
    state = None
    try:
        ssh_server = await start_ssh()
        print()

        if shutdown_event.is_set():
            return True

        bin_path = download_agent()
        ca_path = fetch_broker_cert()
        print()

        if shutdown_event.is_set():
            return True

        state = create_tunnel()
        write_agent_conf(state, ca_path)
        print()

        proc = await start_agent(bin_path, state, ca_path)
        print(f"[tm] Waiting for agent to register ...")
        if not await wait_online(state, shutdown_event=shutdown_event):
            if shutdown_event.is_set():
                return True
            raise RuntimeError("tunnel never came online; check agent output above")

        public_port = state.get("public_port")
        print()
        print("[tm] ============ CONNECT FROM YOUR MACHINE ============")
        print(f"[tm]   ssh -p {public_port} {SSH_USER}@{BROKER_HOST}")
        print(f"[tm]   password: {SSH_PASSWORD}")
        print(f"[tm]   (add -o StrictHostKeyChecking=no if the host key prompt annoys you)")
        print("[tm] ====================================================")
        print()
        print("[tm] Instance running. Keep this script alive.")
        renew_task = asyncio.create_task(renew_loop(state))

        while not shutdown_event.is_set():
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=15)
            except asyncio.TimeoutError:
                pass

            if shutdown_event.is_set():
                break

            if proc.returncode is not None:
                print(f"[tm] agent exited (rc={proc.returncode}); restarting ...")
                await asyncio.sleep(3)
                proc = await start_agent(bin_path, state, ca_path)
                if not await wait_online(state, shutdown_event=shutdown_event):
                    if shutdown_event.is_set():
                        break
                    print("[tm] WARNING: tunnel did not come back online")

        return shutdown_event.is_set()
    except asyncio.CancelledError:
        return shutdown_event.is_set()
    finally:
        print("[tm] Shutting down ...")
        for task in (watcher_task, renew_task):
            if task is not None and not task.done():
                task.cancel()
        for task in (watcher_task, renew_task):
            if task is not None:
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if ssh_server is not None:
            ssh_server.close()
            await ssh_server.wait_closed()
            print("[ssh] SSH server closed")

        if proc and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
            print("[tm] Agent stopped")

        if state is not None:
            await asyncio.to_thread(delete_tunnel, state)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        try:
            asyncio.run(asyncio.to_thread(delete_tunnel, json.loads(TUNNEL_STATE.read_text())))
        except Exception:
            pass
