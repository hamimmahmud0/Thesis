# KaggleManager

A local daemon that lets an AI agent (e.g. Claude Code) start **disposable Kaggle VMs** (GPU T4 or CPU),
run and test code on them over SSH, and shut them down again. It keeps several Kaggle accounts,
checks each account's remaining GPU quota, and automatically picks the best one.

It drives the TunnelMate SSH kernels in `kernels/ssh` (GPU) and `kernels/ssh-cpu` (CPU): each start
pushes the kernel to Kaggle, waits for the tunnel's public port, and verifies an SSH login.

One process serves three things on `127.0.0.1:8765`:

| URL | Purpose |
|---|---|
| `/mcp` | MCP server (streamable-HTTP) that Claude connects to |
| `/` | Settings and monitoring GUI |
| `/api/*` | JSON API used by the GUI |

```
Claude ──MCP──┐                         ┌── kaggle CLI (per-account token) ──> Kaggle
              ├─ KaggleManager daemon ──┤
Browser ─GUI──┘   (accounts, selector,  └── asyncssh ──> TunnelMate broker ──> VM
                   VM lifecycle, state)
```

## Requirements

- Linux with Python 3.10+ (developed on 3.13)
- The `kaggle` CLI (v2.x, supports `KAGGLE_API_TOKEN` and `kaggle quota`) on `PATH`
- Python packages: `pip install -r requirements.txt` (`mcp`, `uvicorn`, `asyncssh`, `pyyaml`)
- One Kaggle API token per account (Kaggle → Settings → API → Create New Token)

## Setup

1. **Add accounts.** Put one token per file in `.tokens/`; the filename is just a label:
   ```
   .tokens/alice      # contains: KGAT_xxxxxxxx...
   .tokens/bob
   ```
   The real Kaggle username is detected automatically (the label and username may differ).
   Accounts can also be added, validated and removed in the GUI.
2. **Run the server:**
   ```
   python server.py
   ```
   GUI: <http://127.0.0.1:8765/>, MCP: <http://127.0.0.1:8765/mcp>
3. **Register with Claude Code:**
   ```
   claude mcp add --transport http kaggle-vm http://127.0.0.1:8765/mcp
   ```

### Autostart at boot (systemd user service)

The unit lives at `~/.config/systemd/user/kaggle-manager.service` and is enabled. User lingering is on,
so it starts at boot without logging in.

```
systemctl --user status  kaggle-manager
systemctl --user restart kaggle-manager        # after changing code
journalctl --user -u kaggle-manager -f         # logs
```

To recreate it on another machine, use a unit with `WorkingDirectory` set to this folder,
`ExecStart=<python3> server.py`, a `PATH` that contains the `kaggle` binary, and `Restart=on-failure`;
then `systemctl --user enable --now kaggle-manager` and `loginctl enable-linger $USER`.

## MCP tools

Typical flow: `start_vm` → `upload` / `run_command` → `download` → `stop_vm`.

| Tool | Arguments | Description |
|---|---|---|
| `list_accounts` | `refresh=false` | Accounts with GPU quota, refresh time, cooldown. Never returns tokens |
| `pick_account` | `kind="gpu"` | Dry run: ranked candidates and why others were rejected |
| `start_vm` | `kind="gpu"\|"cpu"`, `account=""` | Start a VM on the best (or the named) account; takes about 1–2 min. Returns `vm_id`, host, port, ssh command |
| `list_vms` | – | Running VMs started by this server |
| `vm_status` | `vm_id=""` | Kernel status and SSH reachability |
| `run_command` | `command`, `vm_id=""`, `timeout=300`, `cwd=""` | Run a shell command; returns `exit_code`, `stdout`, `stderr` |
| `upload` | `local_path`, `remote_path`, `vm_id=""` | Copy a file or directory to the VM |
| `download` | `remote_path`, `local_path`, `vm_id=""` | Copy a file or directory from the VM |
| `get_logs` | `vm_id=""`, `seconds=10` | Kernel boot/tunnel logs |
| `stop_vm` | `vm_id=""` | Stop the VM and delete its kernel |

`vm_id` may be omitted when exactly one VM is running.

## Account selection

For `start_vm`, accounts are filtered and then ranked.

**Rejected if:** disabled; not allowed for that kind (GPU/CPU flag off); already has a VM of that kind
(Kaggle allows one session per kernel); in cooldown after a failure; quota check failed; or (GPU only)
less than `min_gpu_hours` remaining.

**Ranking:**
- **GPU:** manual `priority` (higher first), then most remaining GPU hours, then soonest quota refresh.
- **CPU:** manual `priority`, then least recently used. CPU time is not metered by `kaggle quota`.

If a start fails (bad token, session limit, boot error) the account goes into cooldown and the next
candidate is tried automatically.

## GUI

- **Dashboard:** quota bars per account, which account is next for GPU, start GPU/CPU VM buttons.
- **Accounts:** add (token is validated first), remove, enable/disable, GPU/CPU permission, priority, test.
- **VMs:** running VMs with uptime/idle time, SSH command, copy-with-password, kernel logs, stop.
- **Settings:** selection policy, safety limits, TunnelMate/template settings.
- **Activity:** live log of selector decisions, VM lifecycle, MCP commands and errors.

Saved settings apply immediately. Changing `server.host` / `server.port` needs a restart.

## Configuration

Settings are stored in `config.yaml` (written by the GUI; defaults are in `kmgr/config.py`).

| Key | Default | Meaning |
|---|---|---|
| `selection.min_gpu_hours` | `1.0` | Skip GPU accounts with less quota left |
| `selection.quota_cache_seconds` | `60` | How long a quota reading is reused |
| `selection.bad_account_seconds` | `900` | Cooldown after a failed start |
| `safety.max_vms` | `3` | Concurrent VM limit |
| `safety.idle_stop_minutes` | `30` | Auto-stop VMs with no commands for this long (`0` = off) |
| `safety.boot_timeout_seconds` | `420` | Give up waiting for a VM to come up |
| `safety.stop_all_on_exit` | `false` | Stop every VM when the server shuts down |
| `vm.gpu_template` / `cpu_template` | `kernels/ssh`, `kernels/ssh-cpu` | Kernel template folders |
| `vm.gpu_slug` / `cpu_slug` | `ssh-gpu`, `ssh-cpu` | Kernel names created under each account |
| `vm.broker`, `broker_host`, `broker_control_port`, `scope`, `protocol` | see file | TunnelMate broker settings |
| `vm.hf_token` | empty | Optional HuggingFace token passed to the VM |
| `accounts.<name>` | – | Per-account `enabled`, `priority`, `gpu`, `cpu`, `username` overrides |

## How a VM start works

1. Build a temporary copy of the template: `push.py` = generated `CONFIG` + the template's `main.py`,
   and `kernel-metadata.json` with `id = <username>/<slug>`. A random SSH password is generated per VM.
2. Delete any stale kernel of that name, then `kaggle kernels push`.
3. Poll kernel status and stream logs until the `ssh -p <port> user@host` line appears.
4. Confirm an actual SSH login, then record the VM in `state.json`.

`stop_vm` touches `/tmp/shutdown_notebook` on the VM (clean tunnel shutdown), then deletes the kernel,
which frees the session slot.

## Project layout

```
server.py          daemon: MCP tools, JSON API, GUI route, idle reaper
static/index.html  single-page GUI (no build step)
kmgr/config.py     settings load/save, token files
kmgr/accounts.py   username and quota cache, cooldowns
kmgr/selector.py   pure ranking logic
kmgr/kaggle_cli.py async wrappers around the kaggle CLI
kmgr/vm.py         VM start/stop, state file
kmgr/ssh_exec.py   SSH exec and SCP
kmgr/events.py     in-memory activity log
.tokens/           API tokens, one file per account   (gitignored)
config.yaml        settings                           (gitignored)
state.json         running VMs incl. SSH passwords    (gitignored)
```

## Security notes

- Binds to `127.0.0.1` only, and rejects requests whose `Host` or `Origin` is not local, which blocks
  DNS-rebinding and cross-site calls from web pages. There is no other authentication, so anyone with
  local access to the machine can use it.
- Tokens are write-only through the API and GUI. Token files and `config.yaml` are mode `0600`.
- VM SSH ports are reachable from the internet through the TunnelMate broker, so each VM gets a random
  password. It is stored in `state.json` and shown in the GUI and in `start_vm` results.
- Keep `.tokens/`, `config.yaml` and `state.json` out of git (already in the repo `.gitignore`).

## Troubleshooting

| Symptom | Check |
|---|---|
| `no usable gpu account: {...}` | The error lists the reason per account (quota, cooldown, VM already running) |
| Account stuck in cooldown | Wait `bad_account_seconds`, or restart the service to clear it |
| Boot timeout | `get_logs` for the VM; the broker may be unreachable, or Kaggle queued the session |
| `kaggle: command not found` under systemd | Make sure the unit's `PATH` contains the `kaggle` binary |
| Quota looks stale | `list_accounts(refresh=true)` or the GUI's "Refresh quota" |
| Orphaned kernel after a crash | `KAGGLE_API_TOKEN=$(cat .tokens/<name>) kaggle kernels delete <user>/ssh-cpu -y` |

## Limitations

- A VM lives in a Kaggle session (limited run time and a weekly GPU quota). Anything not downloaded
  is lost on `stop_vm`.
- CPU usage is not checked against a quota. Only GPU hours are used for ranking.
- `state.json` is the source of truth for running VMs. A VM started outside this server is not tracked.
- GPU and CPU VMs have both been tested end to end (a GPU VM exposes 2× Tesla T4 with CUDA PyTorch).
