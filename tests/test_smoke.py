import tempfile
from pathlib import Path

import torch
from PIL import Image

from maskgit.data import ImageFolderDataset, TokenDataset
from maskgit.models import (MaskGIT, PatchDiscriminator, VQVAE, discriminator_hinge_loss,
                            generate_tokens, perceptual_reconstruction_loss, random_mask)


def test_models_and_data():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        Image.new("RGB", (40, 50), "red").save(root / "one.png")
        sample = ImageFolderDataset(root, 32, False)[0]
        assert sample.shape == (3, 32, 32)
        vq = VQVAE(hidden=32, embedding_dim=16, codebook_size=32)
        reconstruction, codes, qloss = vq(sample.unsqueeze(0))
        assert reconstruction.shape == (1, 3, 32, 32)
        assert codes.shape == (1, 4, 4) and qloss.ndim == 0
        discriminator = PatchDiscriminator(base_channels=16, layers=2)
        real_logits = discriminator(sample.unsqueeze(0))
        fake_logits = discriminator(reconstruction.detach())
        assert real_logits.shape == fake_logits.shape
        assert discriminator_hinge_loss(real_logits, fake_logits).ndim == 0
        assert perceptual_reconstruction_loss(reconstruction, sample.unsqueeze(0)).ndim == 0
        path = root / "tokens.pt"
        torch.save({"tokens": codes.to(torch.int16), "meta": {"grid_size": 4, "codebook_size": 32}}, path)
        assert TokenDataset(path)[0].shape == (4, 4)
        model = MaskGIT(32, 4, dimension=32, layers=1, heads=4, dropout=0)
        masked, mask = random_mask(codes); masked[masked < 0] = model.mask_id
        assert model(masked).shape == (1, 16, 32) and mask.any()
        generated = generate_tokens(model.eval(), 2, steps=3)
        assert generated.shape == (2, 4, 4) and generated.max() < 32
