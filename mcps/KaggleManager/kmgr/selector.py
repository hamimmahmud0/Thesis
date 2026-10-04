"""Pure account-selection logic (no I/O) so it is easy to test."""
import time


def rank(accounts, kind, min_hours, busy=(), bad_until=None, now=None):
    """accounts: list of dicts {name, enabled, priority, gpu, cpu, quota (or None), last_used}.
    Returns (ordered_candidates, rejected: {name: reason})."""
    now = now or time.time()
    bad_until = bad_until or {}
    ok, rejected = [], {}
    for a in accounts:
        n = a["name"]
        if not a.get("enabled", True):
            rejected[n] = "disabled"
        elif not a.get(kind, True):
            rejected[n] = f"{kind} not allowed"
        elif n in busy:
            rejected[n] = f"{kind} VM already running"
        elif bad_until.get(n, 0) > now:
            rejected[n] = "cooling down after failure"
        elif a.get("error"):
            rejected[n] = f"quota check failed: {a['error']}"
        elif kind == "gpu" and (a["quota"].get("GPU", {}).get("remaining", 0) < min_hours):
            rejected[n] = f"GPU quota {a['quota'].get('GPU', {}).get('remaining', 0):.2f}h < {min_hours}h"
        else:
            ok.append(a)
    if kind == "gpu":
        # most remaining hours; manual priority first; soonest refresh breaks ties
        key = lambda a: (-a.get("priority", 0), -a["quota"]["GPU"]["remaining"],
                         a["quota"]["GPU"].get("refresh_at") or "")
    else:
        key = lambda a: (-a.get("priority", 0), a.get("last_used", 0))  # least recently used
    return sorted(ok, key=key), rejected
