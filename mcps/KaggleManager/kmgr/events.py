import collections, time

_events = collections.deque(maxlen=300)


def log(msg, kind="info"):
    _events.append({"ts": time.time(), "kind": kind, "msg": msg})


def recent():
    return list(_events)
