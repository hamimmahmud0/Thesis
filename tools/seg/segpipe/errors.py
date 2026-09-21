from dataclasses import dataclass
class PipelineError(RuntimeError):
    def __init__(self, code: str, message: str): self.code = code; super().__init__(message)
@dataclass(frozen=True)
class Issue: code: str; safe_message: str
def classify_exception(exc: BaseException) -> Issue:
    if isinstance(exc, PipelineError): return Issue(exc.code, str(exc))
    text = str(exc).lower()
    rules = ((("401", "unauthorized", "invalid token"), "invalid_token"),
             (("no space left", "disk quota", "storage limit"), "storage_full"),
             (("cuda out of memory", "cuda oom"), "cuda_oom"),
             (("out of memory", "cannot allocate memory"), "cpu_oom"),
             (("repository not found", "404", "not found"), "source_not_found"))
    for needles, code in rules:
        if any(n in text for n in needles): return Issue(code, f"{type(exc).__name__}: {exc}")
    return Issue("unexpected", f"{type(exc).__name__}: {exc}")
