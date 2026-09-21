import urllib.parse
import urllib.request
class Notifier:
    def __init__(self, token: str = "", chat_id: str = ""): self.token, self.chat_id = token, chat_id
    @classmethod
    def from_config(cls, config): return cls(config.bot_token, config.chat_id)
    def major(self, stage: str, message: str) -> None:
        text = f"[seg/{stage}] {message}"; print(text, flush=True)
        if not self.token or not self.chat_id: return
        data = urllib.parse.urlencode({"chat_id": self.chat_id, "text": text}).encode()
        try:
            request = urllib.request.Request(f"https://api.telegram.org/bot{self.token}/sendMessage", data=data, method="POST")
            with urllib.request.urlopen(request, timeout=10): pass
        except Exception as exc: print(f"warning: notification failed: {type(exc).__name__}", flush=True)
