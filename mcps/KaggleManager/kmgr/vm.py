"""VM lifecycle: pick account -> push kernel -> wait for tunnel port -> SSH-verify."""
import asyncio, json, pprint, re, secrets, shutil, tempfile, time, uuid
from pathlib import Path

from . import accounts, config, kaggle_cli, selector, ssh_exec

_state_lock = asyncio.Lock()
_acct_locks = {}
LOG_RE = re.compile(r"ssh -p (\d+) (\S+)@(\S+)")


def _load():
    try:
        return json.loads(config.STATE_PATH.read_text())
    except Exception:
        return {}


def _save(s):
    config.STATE_PATH.write_text(json.dumps(s, indent=2))
    config.STATE_PATH.chmod(0o600)


def public(vm):
    """VM record without secrets in the ssh_password field name kept for the agent."""
    return dict(vm)


def get(vm_id):
    s = _load()
    if vm_id is None and len(s) == 1:
        return next(iter(s.values()))
    if vm_id not in s:
        raise KeyError(f"unknown vm_id {vm_id!r}; known: {list(s)}")
    return s[vm_id]


def list_vms():
    return list(_load().values())


def touch(vm_id):
    s = _load()
    if vm_id in s:
        s[vm_id]["last_activity"] = time.time()
        _save(s)


def _build(cfg, kind, uname, password):
    v = cfg["vm"]
    tpl = Path(v[f"{kind}_template"])
    slug = v[f"{kind}_slug"]
    tmp = Path(tempfile.mkdtemp(prefix=f"kvm-{kind}-"))
    shutil.copy(tpl / "main.py", tmp / "main.py")
    meta = json.loads((tpl / "kernel-metadata.json").read_text())
    meta["id"] = f"{uname}/{slug}"
    meta["code_file"] = "push.py"
    (tmp / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    conf = {
        "HF_TOKEN": v.get("hf_token", ""), "BUCKET": "kaggle",
        "SSH_USER": v["ssh_user"], "SSH_PASSWORD": password,
        "SSH_HOST": v["ssh_local_host"], "SSH_PORT": v["ssh_local_port"],
        "TUNNELMATE_BROKER": v["broker"], "BROKER_HOST": v["broker_host"],
        "BROKER_CONTROL_PORT": v["broker_control_port"], "SCOPE": v["scope"], "PROTOCOL": v["protocol"],
    }
    (tmp / "push.py").write_text("CONFIG = " + pprint.pformat(conf, sort_dicts=False) + "\n"
                                 + (tpl / "main.py").read_text())
    return tmp, f"{uname}/{slug}"


async def choose(kind, cfg, account=None):
    infos = await accounts.all_info(cfg)
    busy = {v["account"] for v in list_vms() if v["kind"] == kind}
    if account:
        infos = [i for i in infos if i["name"] == account]
        if not infos:
            raise ValueError(f"unknown account {account!r}")
    ranked, rejected = selector.rank(infos, kind, cfg["selection"]["min_gpu_hours"],
                                     busy, accounts.bad_until)
    return ranked, rejected


async def start(kind, cfg=None, account=None, log=lambda m: None):
    cfg = cfg or config.load()
    if kind not in ("gpu", "cpu"):
        raise ValueError("kind must be 'gpu' or 'cpu'")
    if len(list_vms()) >= cfg["safety"]["max_vms"]:
        raise RuntimeError(f"max_vms ({cfg['safety']['max_vms']}) reached; stop a VM first")
    ranked, rejected = await choose(kind, cfg, account)
    if not ranked:
        raise RuntimeError(f"no usable {kind} account: {json.dumps(rejected)}")
    errors = {}
    for cand in ranked:
        name = cand["name"]
        lock = _acct_locks.setdefault(name, asyncio.Lock())
        async with lock:
            log(f"trying account {name}")
            try:
                return await _start_on(kind, cfg, cand, log)
            except Exception as e:
                errors[name] = str(e)
                log(f"account {name} failed: {e}")
                accounts.mark_bad(name, cfg)
    raise RuntimeError(f"all candidate accounts failed: {json.dumps(errors)}")


async def _start_on(kind, cfg, cand, log):
    name = cand["name"]
    token = config.read_token(name)
    uname = await accounts.username(name, cfg)
    password = secrets.token_urlsafe(12)
    tmp, ref = _build(cfg, kind, uname, password)
    try:
        # clear any stale kernel from a previous session so a fresh run starts
        if await kaggle_cli.status(token, ref) != "NOT_FOUND":
            await kaggle_cli.delete(token, ref)
            await asyncio.sleep(3)
        log(f"pushing {ref}")
        await kaggle_cli.push(token, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    accounts.last_used[name] = time.time()
    deadline = time.time() + cfg["safety"]["boot_timeout_seconds"]
    port = host = None
    while time.time() < deadline:
        await asyncio.sleep(10)
        st = await kaggle_cli.status(token, ref)
        log(f"kernel status: {st}")
        if st in ("ERROR", "CANCELACKNOWLEDGED", "COMPLETE", "CANCELED"):
            raise RuntimeError(f"kernel ended during boot: {st}\n{(await kaggle_cli.logs(token, ref))[-500:]}")
        m = LOG_RE.search(await kaggle_cli.logs(token, ref))
        if m:
            port, host = int(m.group(1)), m.group(3)
            break
    if not port:
        await kaggle_cli.delete(token, ref)
        raise RuntimeError("timed out waiting for the tunnel port")
    vm = {"vm_id": f"{kind}-{name}-{uuid.uuid4().hex[:6]}", "kind": kind, "account": name,
          "username": uname, "kernel": ref, "host": host, "port": port,
          "ssh_user": cfg["vm"]["ssh_user"], "ssh_password": password,
          "started_at": time.time(), "last_activity": time.time()}
    for _ in range(12):  # sshd may lag the tunnel
        if await ssh_exec.probe(vm):
            break
        await asyncio.sleep(5)
    else:
        await kaggle_cli.delete(token, ref)
        raise RuntimeError(f"tunnel up at {host}:{port} but SSH login failed")
    async with _state_lock:
        s = _load()
        s[vm["vm_id"]] = vm
        _save(s)
    accounts.invalidate(name)
    return vm


async def stop(vm_id):
    vm = get(vm_id)
    try:
        await ssh_exec.exec_cmd(vm, "touch /tmp/shutdown_notebook", timeout=15)
        await asyncio.sleep(5)
    except Exception:
        pass
    ok, out = await kaggle_cli.delete(config.read_token(vm["account"]), vm["kernel"])
    async with _state_lock:
        s = _load()
        s.pop(vm["vm_id"], None)
        _save(s)
    accounts.invalidate(vm["account"])
    return {"vm_id": vm["vm_id"], "kernel_deleted": ok, "detail": out.strip()[:200]}
