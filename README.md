# MaskGIT Lab

A from-scratch, windowed MaskGIT experiment for training an image generator on a folder of images. It includes:

- a compact VQ-VAE visual tokenizer with optional VQGAN sharpening;
- cached discrete image tokens;
- a bidirectional MaskGIT transformer;
- iterative 8–16 pass generation;
- CUDA mixed-precision training, checkpoints, progress, previews, and safe stopping.

## Start

Double-click `launch.bat`, or run:

```powershell
python app.py
```

If dependencies are missing, install them with `python -m pip install -r requirements.txt`. The app has already been designed around a 12 GB RTX 3060. Training time and quality depend heavily on dataset size and consistency.

## Workflow

1. In **Tokenizer**, choose a folder containing PNG, JPG, WebP, or BMP images.
2. Click **Train full pipeline**. It trains the tokenizer, builds `tokens.pt`, then trains MaskGIT automatically.
3. In **Generate**, the two final checkpoints are filled in automatically. Create images when training completes.

The individual stage buttons are still available when you want to experiment or inspect a stage by itself. **Save project** stores all current settings and model paths in a small JSON file; **Load project** restores them later.

The `runs` folder receives checkpoints and the cache. Generated PNGs go to `generated` by default. `vqvae_latest.pt` and `maskgit_latest.pt` are refreshed after each epoch, so interrupted experiments retain the last completed epoch.

## Practical advice

- Begin with a visually consistent dataset of at least a few thousand images.
- First use 2–5 epochs as a pipeline test, then raise the epoch counts.
- If GPU memory runs out, reduce batch size first.
- If you change resolution, click **Optimize for resolution**. The app uses stronger tokenizer compression at 256px and above so the transformer's token grid remains practical.
- A low tokenizer reconstruction loss and crisp reconstruction previews matter before transformer training.
- **VQGAN sharpening** is enabled by default. It adds multi-scale perceptual and adversarial losses after a short warmup. Turn it off for a faster, more stable baseline; lower **GAN strength** if training becomes unstable.
- This is intentionally compact educational code, not a reproduction of Google's full research training recipe.
