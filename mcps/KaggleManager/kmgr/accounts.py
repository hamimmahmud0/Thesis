"""Account registry: token files + per-account settings, cached quota/username."""
import asyncio, time

from . import config, kaggle_cli

_quota_cache = {}   # name -> (ts, quota|None, error|None)
_username = {}      # name -> username
last_used = {}      # name -> ts
bad_until = {}      # name -> ts


async def username(name, cfg=None):
    cfg = cfg or config.load()
    pinned = config.account_settings(cfg, name)["username"]
    if pinned:
        return pinned
    if name not in _username:
        _username[name] = await kaggle_cli.whoami(config.read_token(name))
    return _username[name]


async def info(name, cfg=None, force=False):
    cfg = cfg or config.load()
    ttl = cfg["selection"]["quota_cache_seconds"]
    s = config.account_settings(cfg, name)
    hit = _quota_cache.get(name)
    if force or not hit or time.time() - hit[0] > ttl:
        try:
            q = await kaggle_cli.quota(config.read_token(name))
            hit = (time.time(), q, None)
        except Exception as e:
            hit = (time.time(), None, str(e))
        _quota_cache[name] = hit
    uname = None
    try:
        uname = await username(name, cfg)
    except Exception:
        pass
    return {"name": name, "username": uname, **{k: s[k] for k in ("enabled", "priority", "gpu", "cpu")},
            "quota": hit[1], "error": hit[2], "quota_checked_at": hit[0],
            "last_used": last_used.get(name, 0),
            "cooldown_until": bad_until.get(name, 0) if bad_until.get(name, 0) > time.time() else 0}


async def all_info(cfg=None, force=False):
    cfg = cfg or config.load()
    return await asyncio.gather(*(info(n, cfg, force) for n in config.token_names()))


def invalidate(name):
    _quota_cache.pop(name, None)


def mark_bad(name, cfg):
    bad_until[name] = time.time() + cfg["selection"]["bad_account_seconds"]
