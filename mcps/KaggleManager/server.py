#!/usr/bin/env python3
"""KaggleManager: local daemon = MCP (streamable-HTTP at /mcp) + settings GUI (/) + JSON API (/api)."""
import asyncio, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse

from kmgr import accounts, config, events, kaggle_cli, ssh_exec, vm

cfg0 = config.load()
HOST, PORT = cfg0["server"]["host"], cfg0["server"]["port"]
ALLOWED = [f"{HOST}:{PORT}", f"127.0.0.1:{PORT}", f"localhost:{PORT}", f"kagglemanager:{PORT}", "kagglemanager"]
mcp = FastMCP(
    "kaggle-vm", host=HOST, port=PORT,
    instructions=("Start disposable Kaggle VMs (GPU T4 or CPU) over SSH to test code. Typical flow: "
                  "start_vm -> upload / run_command -> download -> stop_vm. The best account is chosen "
                  "automatically from remaining quota. Always stop_vm when done; VMs burn quota."),
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=ALLOWED,
        allowed_origins=[f"http://{h}" for h in ALLOWED]))


def _vm_view(v, secret=True):
    d = dict(v)
    d["uptime_min"] = round((time.time() - v["started_at"]) / 60, 1)
    d["idle_min"] = round((time.time() - v["last_activity"]) / 60, 1)
    d["ssh_command"] = f"ssh -p {v['port']} {v['ssh_user']}@{v['host']}"
    if not secret:
        d.pop("ssh_password", None)
    return d


def _acct_view(i):
    g = (i["quota"] or {}).get("GPU")
    return {**i, "gpu_remaining": g and g["remaining"], "gpu_total": g and g["total"],
            "gpu_refresh_at": g and g["refresh_at"]}


# ───────────────────────── MCP tools ─────────────────────────

@mcp.tool()
async def list_accounts(refresh: bool = False) -> list:
    """List Kaggle accounts with GPU quota (hours), refresh time, and cooldown state. No tokens are shown."""
    return [_acct_view(i) for i in await accounts.all_info(force=refresh)]


@mcp.tool()
async def pick_account(kind: str = "gpu") -> dict:
    """Dry run: which account would start_vm choose for kind='gpu'|'cpu', and why others were rejected."""
    ranked, rejected = await vm.choose(kind, config.load())
    return {"ranked": [a["name"] for a in ranked], "rejected": rejected}


@mcp.tool()
async def start_vm(kind: str = "gpu", account: str = "") -> dict:
    """Start a Kaggle VM (kind 'gpu' = T4, 'cpu') on the best available account (or the named one).
    Takes ~1-2 min. Returns vm_id and ssh connection details."""
    events.log(f"MCP start_vm kind={kind} account={account or 'auto'}", "mcp")
    try:
        v = await vm.start(kind, account=account or None, log=lambda m: events.log(m, "vm"))
    except Exception as e:
        events.log(f"start_vm failed: {e}", "error")
        raise
    events.log(f"VM {v['vm_id']} ready at {v['host']}:{v['port']}", "vm")
    return _vm_view(v)


@mcp.tool()
async def list_vms() -> list:
    """List running VMs started by this server."""
    return [_vm_view(v) for v in vm.list_vms()]


@mcp.tool()
async def vm_status(vm_id: str = "") -> dict:
    """Kernel status and SSH reachability of a VM (vm_id optional if only one is running)."""
    v = vm.get(vm_id or None)
    st = await kaggle_cli.status(config.read_token(v["account"]), v["kernel"])
    return {**_vm_view(v), "kernel_status": st, "ssh_ok": await ssh_exec.probe(v)}


@mcp.tool()
async def run_command(command: str, vm_id: str = "", timeout: int = 300, cwd: str = "") -> dict:
    """Run a shell command on the VM over SSH. Returns exit_code, stdout, stderr."""
    v = vm.get(vm_id or None)
    vm.touch(v["vm_id"])
    events.log(f"{v['vm_id']}$ {command[:120]}", "mcp")
    r = await ssh_exec.exec_cmd(v, command, timeout=timeout, cwd=cwd or None)
    vm.touch(v["vm_id"])
    return r


@mcp.tool()
async def upload(local_path: str, remote_path: str, vm_id: str = "") -> str:
    """Copy a local file or directory to the VM."""
    v = vm.get(vm_id or None)
    await ssh_exec.upload(v, local_path, remote_path)
    vm.touch(v["vm_id"])
    return f"uploaded {local_path} -> {v['vm_id']}:{remote_path}"


@mcp.tool()
async def download(remote_path: str, local_path: str, vm_id: str = "") -> str:
    """Copy a file or directory from the VM to the local machine."""
    v = vm.get(vm_id or None)
    await ssh_exec.download(v, remote_path, local_path)
    vm.touch(v["vm_id"])
    return f"downloaded {v['vm_id']}:{remote_path} -> {local_path}"


@mcp.tool()
async def get_logs(vm_id: str = "", seconds: int = 10) -> str:
    """Kernel (boot/tunnel) logs of a VM."""
    v = vm.get(vm_id or None)
    return (await kaggle_cli.logs(config.read_token(v["account"]), v["kernel"], seconds))[-6000:]


@mcp.tool()
async def stop_vm(vm_id: str = "") -> dict:
    """Stop a VM and delete its kernel (frees the session and stops quota use)."""
    events.log(f"stopping {vm_id or '(only vm)'}", "vm")
    return await vm.stop(vm.get(vm_id or None)["vm_id"])


# ───────────────────────── JSON API + GUI ─────────────────────────

def guard(request: Request):
    """Local-only: reject foreign Host/Origin (DNS rebinding / cross-site)."""
    if request.headers.get("host", "") not in ALLOWED:
        raise PermissionError("bad host")
    origin = request.headers.get("origin")
    if origin and origin.split("://", 1)[-1] not in ALLOWED:
        raise PermissionError("bad origin")


def api(path, methods):
    def deco(fn):
        @mcp.custom_route(path, methods=methods)
        async def handler(request: Request):
            try:
                guard(request)
                body = await request.json() if request.method in ("POST", "PUT") and await request.body() else {}
                res = await fn(request, body)
                return JSONResponse(res)
            except PermissionError as e:
                return JSONResponse({"error": str(e)}, 403)
            except Exception as e:
                events.log(f"API {request.url.path}: {e}", "error")
                return JSONResponse({"error": str(e)}, 400)
        return fn
    return deco


def _settings_view(cfg):
    c = json.loads(json.dumps(cfg))
    c["vm"]["hf_token"] = "set" if c["vm"].get("hf_token") else ""
    return c


@mcp.custom_route("/", methods=["GET"])
async def index(request: Request):
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@api("/api/state", ["GET"])
async def api_state(request, body):
    cfg = config.load()
    infos = await accounts.all_info(cfg, force=request.query_params.get("refresh") == "1")
    return {"accounts": [_acct_view(i) for i in infos], "vms": [_vm_view(v) for v in vm.list_vms()],
            "settings": _settings_view(cfg), "events": events.recent()}


@api("/api/pick", ["GET"])
async def api_pick(request, body):
    ranked, rejected = await vm.choose(request.query_params.get("kind", "gpu"), config.load())
    return {"ranked": [a["name"] for a in ranked], "rejected": rejected}


@api("/api/settings", ["PUT"])
async def api_settings(request, body):
    body.pop("accounts", None)
    if body.get("vm", {}).get("hf_token") == "set":
        body["vm"].pop("hf_token")
    return _settings_view(config.update(body))


@api("/api/accounts", ["POST"])
async def api_add_account(request, body):
    name, token = body["name"].strip(), body["token"].strip()
    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("name must be alphanumeric")
    await kaggle_cli.whoami(token)  # validates token
    config.write_token(name, token)
    accounts.invalidate(name)
    events.log(f"account {name} added", "info")
    return {"ok": True}


@api("/api/accounts/update", ["POST"])
async def api_update_account(request, body):
    name = body.pop("name")
    allowed = {k: body[k] for k in ("enabled", "priority", "gpu", "cpu", "username") if k in body}
    cur = config.load()["accounts"].get(name, {})
    config.update({"accounts": {**config.load()["accounts"], name: {**cur, **allowed}}})
    return {"ok": True}


@api("/api/accounts/delete", ["POST"])
async def api_del_account(request, body):
    name = body["name"]
    if any(v["account"] == name for v in vm.list_vms()):
        raise ValueError("account has a running VM")
    config.delete_token(name)
    cfg = config.load()
    cfg["accounts"].pop(name, None)
    config.save(cfg)
    return {"ok": True}


@api("/api/accounts/test", ["POST"])
async def api_test_account(request, body):
    i = await accounts.info(body["name"], force=True)
    return _acct_view(i)


@api("/api/vms/start", ["POST"])
async def api_start(request, body):
    events.log(f"GUI start_vm kind={body.get('kind')}", "vm")
    v = await vm.start(body.get("kind", "gpu"), account=body.get("account") or None,
                       log=lambda m: events.log(m, "vm"))
    return _vm_view(v)


@api("/api/vms/stop", ["POST"])
async def api_stop(request, body):
    return await vm.stop(body["vm_id"])


@api("/api/vms/logs", ["POST"])
async def api_logs(request, body):
    v = vm.get(body["vm_id"])
    return {"logs": await kaggle_cli.logs(config.read_token(v["account"]), v["kernel"], 8)}


# ───────────────────────── background: idle reaper ─────────────────────────

async def reaper():
    while True:
        await asyncio.sleep(60)
        try:
            idle = config.load()["safety"]["idle_stop_minutes"]
            if not idle:
                continue
            for v in vm.list_vms():
                if time.time() - v["last_activity"] > idle * 60:
                    events.log(f"{v['vm_id']} idle > {idle} min: stopping", "vm")
                    await vm.stop(v["vm_id"])
        except Exception as e:
            events.log(f"reaper: {e}", "error")


async def serve():
    app = mcp.streamable_http_app()
    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=PORT, log_level="warning"))
    task = asyncio.create_task(reaper())
    events.log(f"KaggleManager on http://{HOST}:{PORT}  (MCP: /mcp)")
    try:
        await server.serve()
    finally:
        task.cancel()
        if config.load()["safety"]["stop_all_on_exit"]:
            for v in vm.list_vms():
                await vm.stop(v["vm_id"])


if __name__ == "__main__":
    print(f"KaggleManager: GUI http://{HOST}:{PORT}/   MCP http://{HOST}:{PORT}/mcp")
    asyncio.run(serve())
