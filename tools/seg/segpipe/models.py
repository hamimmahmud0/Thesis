def build_model(model, classes: int):
    family = model.family
    if family == "segformer":
        from transformers import SegformerForSemanticSegmentation
        checkpoint = model.extra.get("checkpoint", "nvidia/segformer-b0-finetuned-ade-512-512")
        return SegformerForSemanticSegmentation.from_pretrained(
            checkpoint, num_labels=classes, ignore_mismatched_sizes=True
        )
    import segmentation_models_pytorch as smp
    constructors = {"unet": smp.Unet, "unet++": smp.UnetPlusPlus,
                    "deeplabv3": smp.DeepLabV3, "deeplabv3++": smp.DeepLabV3Plus}
    weights = model.pretrained or "imagenet"
    try:
        return constructors[family](encoder_name=model.encoder, encoder_weights=weights,
                                    in_channels=3, classes=classes, activation=None)
    except (KeyError, ValueError) as exc:
        print(f"warning: pretrained weights unavailable for {model.encoder}: {exc}; using initialization", flush=True)
        return constructors[family](encoder_name=model.encoder, encoder_weights=None,
                                    in_channels=3, classes=classes, activation=None)
