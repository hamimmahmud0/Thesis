import json
from pathlib import Path

import pytest

from segpipe.config import ConfigError, load_config
from segpipe.errors import PipelineError
from segpipe.kaggle import KaggleTokenPool, launch_successor
from segpipe.notify import Notifier
from segpipe.storage import HubStore, RunState


def _config(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        "id: seg-test\n"
        "title: Seg test\n"
        "hf_source: owner/source\n"
        "hf_bucket: owner/output\n"
        "hf_token: env:HF_TOKEN\n"
        "kaggle_tokens: [env:KAGGLE_TOKEN_1]\n"
        "bot_token: env:BOT_TOKEN\n"
        "chat_id: '123'\n"
        "models: [unet]\n"
        f"work_dir: {tmp_path / 'work'}\n"
        + extra,
        encoding="utf-8",
    )
    return path


def test_config_resolves_secrets_without_exposing_repr(monkeypatch, tmp_path):
    secrets = {"HF_TOKEN": "hf-private", "KAGGLE_TOKEN_1": "kg-private", "BOT_TOKEN": "bot-private"}
    for key, value in secrets.items():
        monkeypatch.setenv(key, value)
    config = load_config(_config(tmp_path))
    assert config.hf_token == secrets["HF_TOKEN"]
    assert config.kaggle_tokens == (secrets["KAGGLE_TOKEN_1"],)
    assert config.bot_token == secrets["BOT_TOKEN"]
    rendered = repr(config)
    assert all(value not in rendered for value in secrets.values())


def test_config_rejects_scalar_kaggle_token_collection(tmp_path):
    with pytest.raises(ConfigError, match="must be a list"):
        load_config(_config(tmp_path, "kaggle_tokens: env:KAGGLE_TOKEN_1\n"))


@pytest.mark.parametrize(
    "csv_output,expected",
    [
        ("resource,used,remaining,total,refreshAt\nGPU,30.00h,0.00h,30.00h,soon\n", False),
        ("resource,used,remaining,total,refreshAt\nGPU,29.00h,1.00h,30.00h,soon\n", True),
        ("resource,used,remaining,total,refreshAt\nTPU,0.00h,20.00h,20.00h,soon\n", False),
    ],
)
def test_quota_selection_uses_gpu_remaining_column(monkeypatch, csv_output, expected):
    pool = KaggleTokenPool(("token",))
    monkeypatch.setattr(pool, "_identity", lambda token: "user")

    class Result:
        returncode = 0
        stdout = csv_output
        stderr = ""

    calls = []
    monkeypatch.setattr("segpipe.kaggle.subprocess.run", lambda args, **kwargs: calls.append((args, kwargs)) or Result())
    if expected:
        assert pool.select_available() == "token"
    else:
        with pytest.raises(PipelineError, match="available quota"):
            pool.select_available()
    assert calls[0][0] == ["kaggle", "quota", "-v"]
    assert calls[0][1]["env"]["KAGGLE_API_TOKEN"] == "token"


def test_restore_ignores_only_missing_bucket(monkeypatch, tmp_path):
    store = HubStore("owner/source", "owner/output", "token", tmp_path)

    class Missing:
        returncode = 1
        stdout = ""
        stderr = "Bucket not found"

    monkeypatch.setattr(store, "_run", lambda *args, **kwargs: Missing())
    store.restore()

    class Unauthorized:
        returncode = 1
        stdout = ""
        stderr = "401 unauthorized"

    monkeypatch.setattr(store, "_run", lambda *args, **kwargs: Unauthorized())
    with pytest.raises(PipelineError, match="unauthorized"):
        store.restore()


def test_failure_and_checkpoint_state_survive_round_trip(tmp_path):
    store = HubStore("owner/source", "owner/output", "token", tmp_path)
    store.write_state(RunState(models={"unet": {"status": "training", "epoch": 7}}))
    store.append_failure(type("Issue", (), {"code": "cuda_oom", "safe_message": "CUDA exhausted"})())
    state = store.read_state()
    assert state.models["unet"] == {"status": "training", "epoch": 7}
    assert state.failures[0]["code"] == "cuda_oom"
    assert state.failures[0]["time"].endswith("+00:00")


def test_notifier_failure_is_nonfatal_and_does_not_print_token(monkeypatch, capsys):
    token = "bot-private-token"
    notifier = Notifier(token, "123")
    monkeypatch.setattr("segpipe.notify.urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("offline")))
    notifier.major("checkpoint", "epoch 2 complete")
    output = capsys.readouterr().out
    assert "epoch 2 complete" in output and "notification failed" in output
    assert token not in output


def test_handoff_packages_redacted_config_and_private_metadata(monkeypatch, tmp_path):
    kernel = tmp_path / "kernel"
    kernel.mkdir()
    (kernel / "main.py").write_text("print('main')\n", encoding="utf-8")
    (kernel / "bootstrap.py").write_text("print('bootstrap')\n", encoding="utf-8")
    (kernel / "segpipe-1.0.0-py3-none-any.whl").write_bytes(b"wheel")
    (kernel / "tool").mkdir()
    (kernel / "tool" / "pyproject.toml").write_text("[project]\nname='test'\nversion='1'\n", encoding="utf-8")
    secret = "literal-do-not-package"
    config_path = kernel / "config.yaml"
    config_path.write_text(
        "id: seg-test\ntitle: Seg test\nhf_source: owner/source\nhf_bucket: owner/output\n"
        f"hf_token: {secret}\nkaggle_tokens: [{secret}]\nbot_token: {secret}\n"
        "chat_id: '123'\nmodels: [unet]\n"
        f"work_dir: {tmp_path / 'work'}\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    monkeypatch.setattr(KaggleTokenPool, "_identity", lambda self, token: "safe-user")
    captured = {}

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(args, **kwargs):
        package = Path(args[-1])
        captured["config"] = (package / "config.yaml").read_text(encoding="utf-8")
        captured["metadata"] = json.loads((package / "kernel-metadata.json").read_text(encoding="utf-8"))
        captured["kernel"] = (package / "kernel.py").read_text(encoding="utf-8")
        captured["env_token"] = kwargs["env"]["KAGGLE_API_TOKEN"]
        return Result()

    monkeypatch.setattr("segpipe.kaggle.subprocess.run", fake_run)
    launch_successor(config, "handoff-token")
    assert secret not in captured["config"]
    assert captured["metadata"]["is_private"] is True
    assert captured["metadata"]["enable_gpu"] is True
    assert captured["metadata"]["machine_shape"] == "NvidiaTeslaT4"
    assert secret not in captured["kernel"]
    assert "EMBEDDED_WHEEL_B64" in captured["kernel"]
    assert "segpipe-1.0.0-py3-none-any.whl" in captured["kernel"]
    assert captured["metadata"]["code_file"] == "kernel.py"
    assert captured["env_token"] == "handoff-token"
    info = json.loads((config.work_dir / "artifacts" / "kernel.json").read_text(encoding="utf-8"))
    assert info["kernel"] == "safe-user/seg-test"
