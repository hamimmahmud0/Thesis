# segpipe

`segpipe` is the reusable implementation behind `kernels/seg`. It prepares COCO/SAM
instance annotations for YOLO and semantic segmentation, trains the configured models
on two GPUs, persists every checkpoint to a private Hugging Face Bucket, and supports
Kaggle VM handoff and resume.

The accepted `hf_source` may be either a Hugging Face dataset repository ID or a Bucket
URL such as `https://huggingface.co/buckets/owner/dataset`.

The training entrypoint refuses to run outside Kaggle. Local tests are safe:

```bash
PYTHONPATH=tools/seg python -m pytest tools/seg/tests
```
