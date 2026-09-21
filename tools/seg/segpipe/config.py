from __future__ import annotations
import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

class ConfigError(ValueError): pass
SUPPORTED = {"unet", "unet++", "deeplabv3", "deeplabv3++", "segformer"}

def _load_embedded_secrets() -> None:
    key = os.environ.pop("SEGPIPE_EMBEDDED_KEY", "")
    encrypted = os.environ.pop("SEGPIPE_ENCRYPTED_SECRETS", "")
    if not key and not encrypted: return
    if not key or not encrypted: raise ConfigError("Encrypted credential bundle is incomplete")
    try:
        from cryptography.fernet import Fernet, InvalidToken
        values = json.loads(Fernet(key.encode()).decrypt(encrypted.encode()).decode())
    except Exception as exc:
        raise ConfigError("Encrypted credential bundle could not be decrypted") from exc
    if not isinstance(values, dict): raise ConfigError("Encrypted credential bundle is invalid")
    allowed = {"HF_TOKEN", "BOT_TOKEN", "KAGGLE_API_TOKEN"}
    for name, value in values.items():
        if (name in allowed or name.startswith("KAGGLE_TOKEN_")) and isinstance(value, str) and value:
            os.environ[name] = value

def _secret(value: Any, env_name: str) -> str:
    if os.getenv(env_name): return os.environ[env_name]
    if isinstance(value, str) and value.startswith("env:"):
        secret_name = value[4:]
        result = os.getenv(secret_name, "")
        if result: return result
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(secret_name)
        except Exception: return ""
    if value: return str(value)
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(env_name)
    except Exception: return ""

@dataclass(frozen=True)
class ModelConfig:
    name: str; epochs: int = 100; batch: int = 8; image_size: int = 640
    learning_rate: float = 1e-4; encoder: str = "resnet34"
    pretrained: str | None = "imagenet"; workers: int = 4; save_period: int = 1
    extra: dict[str, Any] = field(default_factory=dict)
    @property
    def family(self) -> str:
        return "yolo" if self.name.lower().startswith("yolo") and self.name.lower().endswith("-seg") else self.name.lower()

@dataclass(frozen=True)
class PipelineConfig:
    path: Path; kernel_id: str; title: str; hf_repo: str; hf_bucket: str
    hf_token: str = field(repr=False)
    kaggle_tokens: tuple[str, ...] = field(repr=False)
    models: tuple[ModelConfig, ...]
    work_dir: Path; log_dir: Path; seed: int = 42; train_ratio: float = .8
    val_ratio: float = .1; vm_budget_minutes: int = 690; handoff_minutes: int = 20
    min_root_free_gb: float = 5; min_work_free_gb: float = 3
    bot_token: str = field(default="", repr=False); chat_id: str = ""

def _parse_model(raw: str | dict[str, Any], defaults: dict[str, Any]) -> ModelConfig:
    item = {"name": raw} if isinstance(raw, str) else dict(raw); item = {**defaults, **item}
    name = str(item.pop("name", "")).lower()
    if not name: raise ConfigError("Each model needs a name")
    family = "yolo" if name.startswith("yolo") and name.endswith("-seg") else name
    if family != "yolo" and family not in SUPPORTED: raise ConfigError(f"Unsupported model: {name}")
    known = {k: item.pop(k) for k in list(item) if k in ModelConfig.__dataclass_fields__ and k not in {"name", "extra"}}
    return ModelConfig(name=name, extra=item, **known)

def load_config(path: Path) -> PipelineConfig:
    _load_embedded_secrets()
    path = path.resolve()
    try: raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc: raise ConfigError(f"Config not found: {path}") from exc
    hf_repo = raw.get("hf_source") or raw.get("HF_SOURCE") or raw.get("hf_repo") or raw.get("HF_REPO")
    hf_bucket = raw.get("hf_bucket") or raw.get("HF_BUCKET") or raw.get("DEST_BUCKET")
    if not hf_repo or not hf_bucket: raise ConfigError("hf_source (or hf_repo) and hf_bucket are required")
    models_raw = raw.get("models") or raw.get("MODELS")
    if not isinstance(models_raw, list) or not models_raw: raise ConfigError("models must be a non-empty list")
    models = tuple(_parse_model(x, raw.get("training_defaults", {})) for x in models_raw)
    if len({m.name for m in models}) != len(models): raise ConfigError("model names must be unique")
    split = raw.get("split", {}); train_ratio = float(split.get("train", .8)); val_ratio = float(split.get("val", .1))
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ConfigError("split train and val must be positive and sum to less than 1")
    raw_tokens = raw.get("kaggle_tokens", raw.get("KAGGLE_TOKENS", []))
    if not isinstance(raw_tokens, list): raise ConfigError("kaggle_tokens must be a list")
    tokens = []
    for index, value in enumerate(raw_tokens, 1):
        token = _secret(value, f"KAGGLE_TOKEN_{index}")
        if token: tokens.append(token)
    single = _secret(raw.get("KAGGLE_API_TOKEN"), "KAGGLE_API_TOKEN")
    if single and single not in tokens: tokens.append(single)
    work = Path(raw.get("work_dir", "/kaggle/working/segmentation"))
    return PipelineConfig(path=path, kernel_id=str(raw.get("id", raw.get("ID", "seg-trainer"))),
        title=str(raw.get("title", raw.get("TITLE", "Segmentation trainer"))), hf_repo=str(hf_repo),
        hf_bucket=str(hf_bucket), hf_token=_secret(raw.get("hf_token", raw.get("HF_TOKEN")), "HF_TOKEN"),
        kaggle_tokens=tuple(tokens), models=models, work_dir=work, log_dir=work / "artifacts" / "logs",
        seed=int(raw.get("seed", 42)), train_ratio=train_ratio, val_ratio=val_ratio,
        vm_budget_minutes=int(raw.get("vm_budget_minutes", 690)), handoff_minutes=int(raw.get("handoff_minutes", 20)),
        min_root_free_gb=float(raw.get("min_root_free_gb", 5)), min_work_free_gb=float(raw.get("min_work_free_gb", 3)),
        bot_token=_secret(raw.get("bot_token", raw.get("BOT_TOKEN")), "BOT_TOKEN"),
        chat_id=str(raw.get("chat_id", raw.get("CHAT_ID", ""))))
