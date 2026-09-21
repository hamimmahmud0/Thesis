import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from segpipe.config import ConfigError, ModelConfig, load_config
from segpipe.errors import PipelineError, classify_exception
from segpipe.kaggle import KaggleTokenPool, _redacted_config
from segpipe.main import _training_command
from segpipe.models import build_model
from segpipe.prepare import _clean_polygon, _rle_polygons, convert_coco, deterministic_split
from segpipe.storage import HubStore, RunState

def config(tmp_path, extra=""):
    path = tmp_path / "config.yaml"
    path.write_text("hf_repo: owner/data\nhf_bucket: owner/out\nmodels: [unet, yolo11n-seg]\n" + extra)
    return path

def test_hf_bucket_source_download(monkeypatch, tmp_path):
    store=HubStore("https://huggingface.co/buckets/owner/source","owner/out","token",tmp_path)
    calls=[]
    class Result: returncode=0; stdout=""; stderr=""
    monkeypatch.setattr(store,"_run",lambda args,required=True:(calls.append(args) or Result()))
    store.download_source(tmp_path/"source")
    assert calls[0][:3] == ["hf","buckets","sync"] and calls[0][3] == "hf://buckets/owner/source"

@pytest.mark.parametrize("source", [
    "https://huggingface.co/buckets/owner/source",
    "hf://buckets/owner/source",
])
def test_hf_bucket_source_forms_use_bucket_sync(monkeypatch, tmp_path, source):
    store = HubStore(source, "owner/out", "token", tmp_path); calls = []
    monkeypatch.setattr(store, "_run", lambda args, required=True: calls.append(args))
    store.download_source(tmp_path / "source")
    assert calls == [["hf", "buckets", "sync", "hf://buckets/owner/source", str(tmp_path / "source")]]

def test_hf_dataset_repo_uses_dataset_download(monkeypatch, tmp_path):
    store = HubStore("owner/source", "owner/out", "token", tmp_path); calls = []
    monkeypatch.setattr(store, "_run", lambda args, required=True: calls.append(args))
    store.download_source(tmp_path / "source")
    assert calls == [["hf", "download", "owner/source", "--repo-type", "dataset", "--local-dir", str(tmp_path / "source")]]

def test_config_and_model_families(tmp_path):
    cfg = load_config(config(tmp_path)); assert [x.family for x in cfg.models] == ["unet", "yolo"]
def test_missing_source_config(tmp_path):
    path = tmp_path / "x"; path.write_text("models: [unet]\n")
    with pytest.raises(ConfigError): load_config(path)
def test_bad_model(tmp_path):
    path = tmp_path / "x"; path.write_text("hf_repo: x/y\nhf_bucket: x/y\nmodels: [bad]\n")
    with pytest.raises(ConfigError): load_config(path)
def test_split_is_deterministic_and_complete():
    a = deterministic_split(list(range(20)), .8, .1, 4); b = deterministic_split(list(range(20)), .8, .1, 4)
    assert a == b and set().union(*a.values()) == set(range(20)) and all(a[x].isdisjoint(a[y]) for x, y in (("train","val"),("train","test"),("val","test")))
def test_invalid_sam_polygons_are_removed():
    assert _clean_polygon([[0, 0, 1, 1]], 10, 10) == []
    assert len(_clean_polygon([[-1, 0, 10, 0, 10, 10]], 10, 10)[0]) == 6

def test_uncompressed_rle_is_decoded_to_normalized_polygons(monkeypatch):
    np = pytest.importorskip("numpy")
    calls = []
    class FakeMaskUtils:
        @staticmethod
        def frPyObjects(rle, height, width):
            calls.append((rle, height, width)); return {"encoded": True}
        @staticmethod
        def decode(rle):
            assert rle == {"encoded": True}
            mask = np.zeros((10, 10), dtype="uint8"); mask[2:8, 3:9] = 1
            return mask
    monkeypatch.setitem(__import__("sys").modules, "pycocotools",
        SimpleNamespace(mask=FakeMaskUtils))
    polygons = _rle_polygons({"size": [10, 10], "counts": [20, 4]}, 10, 10)
    assert calls == [({"size": [10, 10], "counts": [20, 4]}, 10, 10)]
    assert polygons and len(polygons[0]) >= 6
    assert all(0 <= coordinate <= 1 for polygon in polygons for coordinate in polygon)
def test_coco_conversion(tmp_path):
    pytest.importorskip("PIL"); from PIL import Image
    source, target = tmp_path / "source", tmp_path / "target"; source.mkdir()
    images=[]; annotations=[]
    for i in range(10):
        name=f"{i}.jpg"; Image.new("RGB", (10,10)).save(source/name)
        images.append({"id":i,"file_name":name,"width":10,"height":10})
        annotations.append({"id":i,"image_id":i,"category_id":3,"segmentation":[[0,0,9,0,9,9]]})
    (source/"instances.json").write_text(json.dumps({"images":images,"annotations":annotations,"categories":[{"id":3,"name":"object"}]}))
    convert_coco(source,target,.8,.1,3)
    assert len(list((target/"images/train").glob("*"))) == 8
    assert (target/"dataset.yaml").exists() and (target/"labels/test").exists()

def test_coco_conversion_preserves_complete_existing_split(tmp_path):
    pytest.importorskip("PIL"); from PIL import Image
    source, target = tmp_path / "source", tmp_path / "target"
    images, annotations = [], []
    for i, split in enumerate(("train", "train", "val", "test")):
        folder = source / "images" / split; folder.mkdir(parents=True, exist_ok=True)
        name = f"{i}.jpg"; Image.new("RGB", (10, 10)).save(folder / name)
        images.append({"id": i, "file_name": f"images/{split}/{name}", "width": 10, "height": 10})
        annotations.append({"id": i, "image_id": i, "category_id": 3, "segmentation": [[0, 0, 9, 0, 9, 9]]})
    (source / "instances.json").write_text(json.dumps({"images": images, "annotations": annotations,
        "categories": [{"id": 3, "name": "object"}]}))
    convert_coco(source, target, .25, .25, 3)
    assert {split: sorted(p.name for p in (target / "images" / split).glob("*"))
            for split in ("train", "val", "test")} == {
                "train": ["0.jpg", "1.jpg"], "val": ["2.jpg"], "test": ["3.jpg"]}

def test_incomplete_existing_split_is_recreated(tmp_path):
    pytest.importorskip("PIL"); from PIL import Image
    source, target = tmp_path / "source", tmp_path / "target"
    images = []
    for i, split in enumerate(("train", "train", "val", "test", None)):
        folder = source / "images" / split if split else source / "images"
        folder.mkdir(parents=True, exist_ok=True); name = f"{i}.jpg"
        Image.new("RGB", (10, 10)).save(folder / name)
        images.append({"id": i, "file_name": name, "width": 10, "height": 10})
    (source / "instances.json").write_text(json.dumps({"images": images, "annotations": [],
        "categories": [{"id": 3, "name": "object"}]}))
    convert_coco(source, target, .6, .2, 7)
    assert [len(list((target / "images" / split).glob("*"))) for split in ("train", "val", "test")] == [3, 1, 1]

def test_trainer_command_dispatches_two_gpu_semantic_and_yolo(tmp_path):
    cfg = SimpleNamespace(path=tmp_path / "config.yaml")
    semantic = _training_command(cfg, ModelConfig("unet"), tmp_path / "dataset", 123.0)
    yolo = _training_command(cfg, ModelConfig("yolo11n-seg"), tmp_path / "dataset", 123.0)
    assert semantic[:3] == ["torchrun", "--standalone", "--nproc_per_node=2"]
    assert semantic[3:5] == ["-m", "segpipe.train_semantic"]
    assert yolo[1:3] == ["-m", "segpipe.train_yolo"] and "torchrun" not in yolo

def test_semantic_model_factory_uses_pretrained_encoder(monkeypatch):
    calls = []
    class FakeSmp:
        Unet = staticmethod(lambda **kwargs: calls.append(kwargs) or kwargs)
        UnetPlusPlus = DeepLabV3 = DeepLabV3Plus = Unet
    monkeypatch.setitem(__import__("sys").modules, "segmentation_models_pytorch", FakeSmp)
    result = build_model(ModelConfig("unet", encoder="resnet50", pretrained="imagenet"), 4)
    assert result["encoder_name"] == "resnet50" and result["encoder_weights"] == "imagenet"
    assert result["classes"] == 4 and result["activation"] is None

def test_segformer_factory_uses_configured_checkpoint(monkeypatch):
    calls = []
    class FakeSegformer:
        @staticmethod
        def from_pretrained(checkpoint, **kwargs):
            calls.append((checkpoint, kwargs)); return "model"
    monkeypatch.setitem(__import__("sys").modules, "transformers",
        SimpleNamespace(SegformerForSemanticSegmentation=FakeSegformer))
    model = build_model(ModelConfig("segformer", extra={"checkpoint": "owner/custom"}), 6)
    assert model == "model" and calls == [("owner/custom", {"num_labels": 6, "ignore_mismatched_sizes": True})]
@pytest.mark.parametrize("message,code", [("401 unauthorized","invalid_token"),("No space left on device","storage_full"),("CUDA out of memory","cuda_oom"),("cannot allocate memory","cpu_oom"),("Repository not found","source_not_found")])
def test_error_classes(message, code): assert classify_exception(RuntimeError(message)).code == code
def test_state_round_trip(tmp_path):
    store=HubStore("a/b","a/c","token",tmp_path); store.update_model("unet",status="complete",epoch=3)
    assert store.read_state().is_complete("unet") and store.read_state().all_complete(["unet"])
def test_hf_invalid_token(monkeypatch,tmp_path):
    store=HubStore("a/b","a/c","",tmp_path)
    with pytest.raises(PipelineError, match="missing"): store.ensure_private_destination()
def test_destination_bucket_is_explicitly_private(monkeypatch,tmp_path):
    store=HubStore("a/b","a/c","token",tmp_path); calls=[]
    monkeypatch.setattr(store,"_run",lambda args,required=True:calls.append(args))
    store.ensure_private_destination()
    assert "--private" in calls[0] and "--exist-ok" in calls[0]
def test_redaction(tmp_path):
    source=tmp_path/"in"; out=tmp_path/"out"
    source.write_text("hf_token: secret\nbot_token: botsecret\nkaggle_tokens: [one, two]\n")
    _redacted_config(source,out); text=out.read_text()
    assert "secret" not in text and "one" not in text and "two" not in text
def test_env_secret_reference_is_not_literal(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cfg=load_config(config(tmp_path,"hf_token: env:DOES_NOT_EXIST\n"))
    assert cfg.hf_token == ""
def test_token_pool_skips_invalid_and_no_quota(monkeypatch):
    pool=KaggleTokenPool(("bad","good")); monkeypatch.setattr(pool,"_identity",lambda token: (_ for _ in ()).throw(PipelineError("bad","bad")) if token=="bad" else "user")
    class Result: returncode=0; stdout="GPU quota available"; stderr=""
    monkeypatch.setattr("segpipe.kaggle.subprocess.run",lambda *a,**k:Result())
    assert pool.select_available()=="good"

def test_complete_pipeline_dry_run(monkeypatch, tmp_path):
    from segpipe.main import run
    path=config(tmp_path, f"work_dir: {tmp_path / 'work'}\nmin_root_free_gb: 0\nmin_work_free_gb: 0\n")
    monkeypatch.setattr("segpipe.storage.HubStore.ensure_private_destination",lambda self:None)
    monkeypatch.setattr("segpipe.storage.HubStore.restore",lambda self:None)
    assert run(path, allow_local=True, dry_run=True) == 0

def test_pipeline_refuses_local_training(tmp_path):
    from segpipe.main import run
    with pytest.raises(ConfigError, match="Kaggle"): run(config(tmp_path))
