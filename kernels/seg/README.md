# Resumable segmentation trainer

This directory is a private Kaggle pipeline for sequentially training YOLO segmentation,
U-Net, U-Net++, DeepLabV3, DeepLabV3+, and SegFormer models on a COCO-format dataset.
It uses both T4 GPUs, uploads every epoch checkpoint and detailed logs to a private
Hugging Face Bucket, and launches a successor kernel when its time window is nearly over.

## Configure and launch

1. Copy `config.yaml.example` to the ignored `config.yaml` and fill in `hf_source`,
   `hf_bucket`, model parameters, and the Telegram chat ID.
2. Store `HF_TOKEN`, `BOT_TOKEN`, and `KAGGLE_TOKEN_1` etc. as environment variables
   locally and as Kaggle Secrets for every Kaggle account that may run a successor.
3. Run `./push` once locally. Training itself is guarded so it cannot run locally.

The pushed configuration contains only `env:...` references. Tokens are used for
authentication but are never written to kernel metadata, source, state, or logs.

Run local tests from the repository root with
`PYTHONPATH=tools/seg python -m pytest tools/seg/tests`. They use synthetic data and mocks; they do
not start training or contact Kaggle/Hugging Face.
