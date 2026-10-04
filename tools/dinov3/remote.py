#!/usr/bin/env python3
"""Run code on the remote GPU box; files are synced through a Hugging Face bucket.

  remote.py push            local dir -> bucket
  remote.py pull            bucket -> remote_dir (on the remote)
  remote.py run <cmd...>    push, pull, then run <cmd> in remote_dir
  remote.py fetch <path>    remote file/dir -> bucket -> ./outputs/
Credentials are read from config.yaml (gitignored).
"""
import os, shlex, subprocess, sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
cfg = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
hf, ssh = cfg["hf"], cfg["ssh"]
BUCKET = f"hf://buckets/{hf['bucket']}"
EXCLUDES = ["webapp/submissions.log", "webapp/public/inspect/*", "webapp/public/b1/*", "config.yaml", ".git/*", "__pycache__/*", "outputs/*", "*.pyc"]
excl = " ".join(f"--exclude {shlex.quote(e)}" for e in EXCLUDES)


def remote(cmd, token=None):
    """Run cmd on the remote; HF token is passed on stdin, not on the command line."""
    script = f"export HF_TOKEN=$(head -n1); {cmd}"
    argv = ["sshpass", "-e", "ssh", "-p", str(ssh["port"]), "-o", "StrictHostKeyChecking=accept-new",
            "-o", "LogLevel=ERROR", f"{ssh['user']}@{ssh['host']}", f"bash -c {shlex.quote(script)}"]
    env = {**os.environ, "SSHPASS": ssh["password"]}
    return subprocess.run(argv, input=(token or hf["token"]) + "\n", text=True, env=env).returncode


def push():
    env = {**os.environ, "HF_TOKEN": hf["token"]}
    return subprocess.run(f"hf buckets sync {shlex.quote(HERE)} {BUCKET} --delete {excl}",
                          shell=True, env=env).returncode


def pull():
    d = ssh["remote_dir"]
    return remote(f"mkdir -p {d} && hf buckets sync {BUCKET} {d} --delete {excl}")


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    if a[0] == "push":
        sys.exit(push())
    if a[0] == "pull":
        sys.exit(pull())
    if a[0] == "model":  # gated weights: uses hf.model_token, only for this download
        sys.exit(remote("hf download facebook/dinov3-vit7b16-pretrain-sat493m --local-dir /root/work/model",
                        token=hf["model_token"]))
    if a[0] == "run":
        if push() or pull():
            sys.exit(1)
        sys.exit(remote(f"cd {ssh['remote_dir']} && {' '.join(a[1:])}"))
    if a[0] == "fetch":
        p = a[1]
        rc = remote(f"hf buckets cp {shlex.quote(p)} {BUCKET}/outputs/$(basename {shlex.quote(p)})")
        if rc:
            sys.exit(rc)
        os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
        env = {**os.environ, "HF_TOKEN": hf["token"]}
        sys.exit(subprocess.run(f"hf buckets sync {BUCKET}/outputs {shlex.quote(os.path.join(HERE,'outputs'))}",
                                shell=True, env=env).returncode)
    sys.exit(__doc__)


main()
