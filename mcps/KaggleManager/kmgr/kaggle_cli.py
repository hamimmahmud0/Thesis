"""Async wrappers over the `kaggle` CLI; the token is passed per call via env."""
import asyncio, json, os, re


class KaggleError(RuntimeError):
    pass


async def run(token, *args, cwd=None, timeout=120):
    env = {**os.environ, "KAGGLE_API_TOKEN": token}
    env.pop("KAGGLE_USERNAME", None)
    env.pop("KAGGLE_KEY", None)
    proc = await asyncio.create_subprocess_exec(
        "kaggle", *args, cwd=cwd, env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise KaggleError(f"kaggle {' '.join(args)} timed out after {timeout}s")
    return proc.returncode, out.decode("utf-8", "replace")


async def whoami(token):
    rc, out = await run(token, "config", "view", timeout=40)
    m = re.search(r"^- username:\s*(\S+)", out, re.M)
    if rc or not m:
        raise KaggleError(f"cannot determine username: {out.strip()[:200]}")
    return m.group(1)


def _hours(s):
    return float(str(s).rstrip("h"))


async def quota(token):
    """-> {'GPU': {'used','remaining','total','refresh_at'}, 'TPU': {...}} in hours."""
    rc, out = await run(token, "quota", "--format", "json", timeout=40)
    try:
        rows = json.loads(out)
    except ValueError:
        raise KaggleError(f"quota failed: {out.strip()[:200]}")
    return {r["resource"]: {"used": _hours(r["used"]), "remaining": _hours(r["remaining"]),
                            "total": _hours(r["total"]), "refresh_at": r.get("refreshAt")}
            for r in rows}


async def push(token, folder):
    rc, out = await run(token, "kernels", "push", "-p", str(folder), timeout=180)
    if rc or "error" in out.lower() and "successfully" not in out.lower():
        raise KaggleError(f"push failed: {out.strip()[-400:]}")
    return out


async def status(token, ref):
    """-> RUNNING / COMPLETE / ERROR / QUEUED / CANCEL... or 'NOT_FOUND'."""
    rc, out = await run(token, "kernels", "status", ref, timeout=40)
    m = re.search(r'status\s+"?(?:KernelWorkerStatus\.)?([A-Za-z_]+)"?', out)
    if m:
        return m.group(1).upper()
    if "denied" in out.lower() or "not found" in out.lower() or "404" in out:
        return "NOT_FOUND"
    raise KaggleError(f"status unparseable: {out.strip()[:200]}")


async def logs(token, ref, seconds=12):
    """Running kernels only expose logs via `-f`; stream briefly and return what arrived."""
    env = {**os.environ, "KAGGLE_API_TOKEN": token}
    proc = await asyncio.create_subprocess_exec(
        "kaggle", "kernels", "logs", "-f", ref, env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    buf = bytearray()

    async def pump():
        while chunk := await proc.stdout.read(4096):
            buf.extend(chunk)

    task = asyncio.create_task(pump())
    try:
        await asyncio.wait_for(asyncio.shield(task), seconds)
    except asyncio.TimeoutError:
        pass
    finally:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()
        task.cancel()
    return buf.decode("utf-8", "replace")


async def delete(token, ref):
    rc, out = await run(token, "kernels", "delete", ref, "-y", timeout=60)
    return rc == 0, out
