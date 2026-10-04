"""SSH/SFTP access to a running VM."""
import asyncio
from pathlib import Path

import asyncssh


async def connect(vm, timeout=20):
    return await asyncio.wait_for(asyncssh.connect(
        vm["host"], vm["port"], username=vm["ssh_user"], password=vm["ssh_password"],
        known_hosts=None, keepalive_interval=20), timeout)


async def exec_cmd(vm, cmd, timeout=300, cwd=None):
    async with await connect(vm) as conn:
        full = f"cd {cwd} && {cmd}" if cwd else cmd
        try:
            r = await asyncio.wait_for(conn.run(full, check=False, term_type=None), timeout)
        except asyncio.TimeoutError:
            return {"exit_code": None, "stdout": "", "stderr": f"timed out after {timeout}s"}
        noise = ("cannot set terminal process group", "no job control in this shell")
        err = "\n".join(l for l in (r.stderr or "").splitlines() if not any(n in l for n in noise))
        return {"exit_code": r.exit_status, "stdout": r.stdout or "", "stderr": err}


async def upload(vm, local, remote):
    async with await connect(vm) as conn:
        await asyncssh.scp(str(Path(local).expanduser()), (conn, remote), recurse=True)


async def download(vm, remote, local):
    async with await connect(vm) as conn:
        await asyncssh.scp((conn, remote), str(Path(local).expanduser()), recurse=True)


async def probe(vm):
    try:
        r = await exec_cmd(vm, "echo ok", timeout=20)
        return r["stdout"].strip() == "ok"
    except Exception:
        return False
