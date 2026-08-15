from __future__ import annotations

import time
from pathlib import Path
from threading import Event
from typing import Callable

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import ImageFolderDataset, TokenDataset, find_images, prepare_image
from .models import (MaskGIT, PatchDiscriminator, VQVAE, discriminator_hinge_loss,
                     generate_tokens, perceptual_reconstruction_loss, random_mask)

Reporter = Callable[[str, dict], None]


def device_for(use_cuda: bool = True) -> torch.device:
    return torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")


def save_checkpoint(path: str | Path, model, **extra) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": model.config, **extra}, path)


def load_vqvae(path: str | Path, device: torch.device) -> tuple[VQVAE, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = VQVAE(**payload["config"]).to(device)
    model.load_state_dict(payload["model"])
    return model.eval(), payload


def load_maskgit(path: str | Path, device: torch.device) -> tuple[MaskGIT, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = MaskGIT(**payload["config"]).to(device)
    model.load_state_dict(payload["model"])
    return model.eval(), payload


def train_vqvae(config: dict, report: Reporter, stop: Event) -> str:
    device = device_for(config.get("cuda", True))
    dataset = ImageFolderDataset(config["dataset"], config["image_size"], augment=True)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                        num_workers=0, pin_memory=device.type == "cuda", drop_last=len(dataset) >= config["batch_size"])
    model = VQVAE(config["hidden"], config["embedding_dim"], config["codebook_size"],
                  config.get("downsample_factor", 8)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    use_vqgan = bool(config.get("vqgan", False))
    discriminator = PatchDiscriminator(config.get("discriminator_channels", 64)).to(device) if use_vqgan else None
    discriminator_optimizer = (torch.optim.AdamW(discriminator.parameters(), lr=config["learning_rate"],
                                                  betas=(0.5, 0.9)) if discriminator else None)
    discriminator_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    adversarial_weight = float(config.get("adversarial_weight", 0.1))
    discriminator_start_epoch = int(config.get("discriminator_start_epoch", 2))
    output = Path(config["output"]); output.mkdir(parents=True, exist_ok=True)
    step = 0; started = time.time()
    mode = "VQGAN" if use_vqgan else "VQ-VAE"
    report("log", {"text": f"Training {mode} tokenizer on {len(dataset)} images using {device}."})
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for images in loader:
            if stop.is_set():
                report("log", {"text": "Tokenizer training stopped."}); return "stopped"
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                reconstruction, _, q_loss = model(images)
                reconstruction_loss = F.l1_loss(reconstruction, images)
                perceptual_loss = perceptual_reconstruction_loss(reconstruction, images) if use_vqgan else reconstruction_loss.new_zeros(())
                adversarial_loss = reconstruction_loss.new_zeros(())
                adversarial_active = discriminator is not None and epoch >= discriminator_start_epoch
                if adversarial_active:
                    discriminator_optimizer.zero_grad(set_to_none=True)
                    real_logits = discriminator(images.detach())
                    fake_logits = discriminator(reconstruction.detach())
                    discriminator_loss = discriminator_hinge_loss(real_logits, fake_logits)
                    discriminator_scaler.scale(discriminator_loss).backward()
                    discriminator_scaler.step(discriminator_optimizer)
                    discriminator_scaler.update()
                    for parameter in discriminator.parameters(): parameter.requires_grad_(False)
                    adversarial_loss = -discriminator(reconstruction).mean()
                loss = reconstruction_loss + q_loss + 0.5 * perceptual_loss + adversarial_weight * adversarial_loss
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); step += 1
            if adversarial_active:
                for parameter in discriminator.parameters(): parameter.requires_grad_(True)
            if step % 10 == 0 or step == 1:
                report("progress", {"value": (epoch - 1 + (step % max(len(loader), 1)) / max(len(loader), 1)) / config["epochs"],
                       "text": f"Epoch {epoch}/{config['epochs']}  loss {loss.item():.4f}  recon {reconstruction_loss.item():.4f}" +
                               (f"  perceptual {perceptual_loss.item():.4f}" if use_vqgan else "")})
        save_checkpoint(output / "vqvae_latest.pt", model, image_size=config["image_size"], epoch=epoch)
        preview = torch.cat([images[:4], reconstruction[:4]], dim=0)
        report("preview_tensor", {"tensor": preview})
    save_checkpoint(output / "vqvae_final.pt", model, image_size=config["image_size"], epoch=config["epochs"])
    report("log", {"text": f"Tokenizer finished in {(time.time()-started)/60:.1f} minutes."})
    return str(output / "vqvae_final.pt")


@torch.inference_mode()
def build_token_cache(config: dict, report: Reporter, stop: Event) -> str:
    device = device_for(config.get("cuda", True))
    model, payload = load_vqvae(config["vqvae"], device)
    paths = find_images(config["dataset"])
    if not paths: raise ValueError("The dataset contains no supported images.")
    image_size = int(payload.get("image_size", config.get("image_size", 128)))
    batches = []
    report("log", {"text": f"Encoding {len(paths)} images into discrete tokens."})
    for start in range(0, len(paths), config["batch_size"]):
        if stop.is_set(): return "stopped"
        selected = paths[start:start + config["batch_size"]]
        images = torch.stack([prepare_image(p, image_size) for p in selected]).to(device)
        batches.append(model.encode(images).cpu().to(torch.int16))
        report("progress", {"value": min(start + len(selected), len(paths)) / len(paths),
                            "text": f"Encoded {min(start + len(selected), len(paths))}/{len(paths)} images"})
    tokens = torch.cat(batches)
    target = Path(config["cache"]); target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tokens": tokens, "meta": {"image_size": image_size, "grid_size": tokens.shape[-1],
                "codebook_size": model.config["codebook_size"], "count": len(paths)}}, target)
    report("log", {"text": f"Token cache saved to {target}."})
    return str(target)


def train_maskgit(config: dict, report: Reporter, stop: Event) -> str:
    device = device_for(config.get("cuda", True)); dataset = TokenDataset(config["cache"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0,
                        pin_memory=device.type == "cuda", drop_last=len(dataset) >= config["batch_size"])
    meta = dataset.meta
    model = MaskGIT(meta["codebook_size"], meta["grid_size"], config["dimension"],
                    config["layers"], config["heads"], config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output = Path(config["output"]); output.mkdir(parents=True, exist_ok=True)
    step = 0; report("log", {"text": f"Training MaskGIT on {len(dataset)} token grids using {device}."})
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for tokens in loader:
            if stop.is_set(): report("log", {"text": "MaskGIT training stopped."}); return "stopped"
            tokens = tokens.long().to(device); masked, mask = random_mask(tokens); masked[masked < 0] = model.mask_id
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(masked)
                targets = tokens.flatten(1)
                loss = F.cross_entropy(logits[mask], targets[mask])
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); step += 1
            if step % 10 == 0 or step == 1:
                accuracy = (logits[mask].argmax(-1) == targets[mask]).float().mean().item()
                report("progress", {"value": (epoch - 1 + (step % max(len(loader), 1)) / max(len(loader), 1)) / config["epochs"],
                                    "text": f"Epoch {epoch}/{config['epochs']}  loss {loss.item():.4f}  masked accuracy {accuracy:.1%}"})
        save_checkpoint(output / "maskgit_latest.pt", model, epoch=epoch, token_meta=meta)
    save_checkpoint(output / "maskgit_final.pt", model, epoch=config["epochs"], token_meta=meta)
    report("log", {"text": "MaskGIT training complete."})
    return str(output / "maskgit_final.pt")


@torch.inference_mode()
def generate_images(config: dict, report: Reporter, stop: Event):
    device = device_for(config.get("cuda", True))
    vqvae, vp = load_vqvae(config["vqvae"], device); model, _ = load_maskgit(config["maskgit"], device)
    def progress(current, total):
        report("progress", {"value": current / total, "text": f"Refinement pass {current}/{total}"})
    tokens = generate_tokens(model, config["count"], config["steps"], config["temperature"], callback=progress)
    images = vqvae.decode(tokens).cpu()
    report("preview_tensor", {"tensor": images})
    output = Path(config["output"]); output.mkdir(parents=True, exist_ok=True)
    from .data import tensor_to_pil
    stamp = time.strftime("%Y%m%d-%H%M%S")
    files = []
    for index, image in enumerate(images):
        path = output / f"maskgit-{stamp}-{index+1:02d}.png"; tensor_to_pil(image).save(path); files.append(str(path))
    report("log", {"text": f"Saved {len(files)} generated images to {output}."})
    return files


def train_full_pipeline(config: dict, report: Reporter, stop: Event) -> dict:
    """Run the entire MaskGIT workflow without requiring UI hand-offs."""
    def stage_report(start: float, span: float, stage: str):
        def wrapped(kind: str, data: dict):
            if kind == "progress":
                data = {**data, "value": start + span * data["value"], "text": f"{stage}: {data['text']}"}
            report(kind, data)
        return wrapped

    report("log", {"text": "Automatic pipeline: tokenizer → token cache → MaskGIT."})
    vqvae = train_vqvae({**config, "batch_size": config["vq_batch_size"], "epochs": config["vq_epochs"],
                         "learning_rate": config["vq_learning_rate"]}, stage_report(0.0, 0.48, "Tokenizer"), stop)
    if vqvae == "stopped": return {"status": "stopped"}
    cache_config = {**config, "vqvae": vqvae, "cache": config["cache"], "batch_size": config["vq_batch_size"]}
    cache = build_token_cache(cache_config, stage_report(0.48, 0.08, "Token cache"), stop)
    if cache == "stopped": return {"status": "stopped", "vqvae": vqvae}
    maskgit = train_maskgit({**config, "cache": cache, "batch_size": config["mg_batch_size"],
                             "epochs": config["mg_epochs"], "learning_rate": config["mg_learning_rate"]},
                            stage_report(0.56, 0.44, "MaskGIT"), stop)
    if maskgit == "stopped": return {"status": "stopped", "vqvae": vqvae, "cache": cache}
    return {"status": "complete", "vqvae": vqvae, "cache": cache, "maskgit": maskgit}
