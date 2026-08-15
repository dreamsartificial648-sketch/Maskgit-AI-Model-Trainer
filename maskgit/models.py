from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class VectorQuantizer(nn.Module):
    def __init__(self, codes: int, dimension: int, commitment: float = 0.25):
        super().__init__()
        self.codes = codes
        self.dimension = dimension
        self.commitment = commitment
        self.embedding = nn.Embedding(codes, dimension)
        nn.init.uniform_(self.embedding.weight, -1 / codes, 1 / codes)

    def forward(self, z: torch.Tensor):
        flat = z.permute(0, 2, 3, 1).contiguous().view(-1, self.dimension)
        weight = self.embedding.weight
        distance = flat.square().sum(1, keepdim=True) + weight.square().sum(1) - 2 * flat @ weight.t()
        indices = distance.argmin(1)
        quantized = self.embedding(indices).view(z.shape[0], z.shape[2], z.shape[3], self.dimension)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(quantized.detach(), z)
        loss = codebook_loss + self.commitment * commitment_loss
        quantized = z + (quantized - z).detach()
        return quantized, indices.view(z.shape[0], z.shape[2], z.shape[3]), loss

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        values = self.embedding(indices)
        return values.permute(0, 3, 1, 2).contiguous()


class VQVAE(nn.Module):
    """A compact tokenizer with an adjustable spatial compression ratio."""

    def __init__(self, hidden: int = 128, embedding_dim: int = 128, codebook_size: int = 512,
                 downsample_factor: int = 8):
        super().__init__()
        if downsample_factor < 4 or downsample_factor & (downsample_factor - 1):
            raise ValueError("downsample_factor must be a power of two of at least 4")
        levels = int(math.log2(downsample_factor))
        self.config = {"hidden": hidden, "embedding_dim": embedding_dim, "codebook_size": codebook_size,
                       "downsample_factor": downsample_factor}
        first = max(hidden // 2, 32)
        encoder_layers: list[nn.Module] = []
        current = 3
        for level in range(levels):
            target = first if level == 0 else hidden
            encoder_layers += [nn.Conv2d(current, target, 4, 2, 1), nn.SiLU()]
            if level > 0:
                encoder_layers.append(ResidualBlock(target))
            current = target
        encoder_layers += [nn.GroupNorm(min(8, hidden), hidden), nn.SiLU(), nn.Conv2d(hidden, embedding_dim, 1)]
        self.encoder = nn.Sequential(*encoder_layers)
        self.quantizer = VectorQuantizer(codebook_size, embedding_dim)
        decoder_layers: list[nn.Module] = [nn.Conv2d(embedding_dim, hidden, 3, padding=1), ResidualBlock(hidden)]
        targets = [hidden] * (levels - 2) + [first, 3]
        current = hidden
        for target in targets:
            decoder_layers.append(nn.ConvTranspose2d(current, target, 4, 2, 1))
            if target == 3:
                decoder_layers.append(nn.Tanh())
            else:
                decoder_layers += [nn.SiLU(), ResidualBlock(target)]
            current = target
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        _, indices, _ = self.quantizer(self.encoder(images))
        return indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.quantizer.lookup(indices))

    def forward(self, images: torch.Tensor):
        quantized, indices, quantizer_loss = self.quantizer(self.encoder(images))
        return self.decoder(quantized), indices, quantizer_loss


class PatchDiscriminator(nn.Module):
    """Small PatchGAN discriminator used only while training a VQGAN tokenizer."""

    def __init__(self, base_channels: int = 64, layers: int = 3):
        super().__init__()
        blocks: list[nn.Module] = [nn.Conv2d(3, base_channels, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        channels = base_channels
        for level in range(1, layers):
            next_channels = min(base_channels * (2 ** level), 512)
            blocks += [nn.Conv2d(channels, next_channels, 4, 2, 1),
                       nn.GroupNorm(min(8, next_channels), next_channels), nn.LeakyReLU(0.2, inplace=True)]
            channels = next_channels
        blocks.append(nn.Conv2d(channels, 1, 3, padding=1))
        self.net = nn.Sequential(*blocks)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


def discriminator_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return 0.5 * (F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean())


def perceptual_reconstruction_loss(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Multi-scale color and edge loss that does not require downloaded model weights."""
    loss = reconstruction.new_zeros(())
    current_reconstruction, current_target = reconstruction, target
    for scale in range(3):
        loss = loss + F.l1_loss(current_reconstruction, current_target) / (2 ** scale)
        dx_reconstruction = current_reconstruction[..., :, 1:] - current_reconstruction[..., :, :-1]
        dx_target = current_target[..., :, 1:] - current_target[..., :, :-1]
        dy_reconstruction = current_reconstruction[..., 1:, :] - current_reconstruction[..., :-1, :]
        dy_target = current_target[..., 1:, :] - current_target[..., :-1, :]
        loss = loss + 0.25 * (F.l1_loss(dx_reconstruction, dx_target) +
                              F.l1_loss(dy_reconstruction, dy_target)) / (2 ** scale)
        if scale < 2 and min(current_target.shape[-2:]) >= 4:
            current_reconstruction = F.avg_pool2d(current_reconstruction, 2)
            current_target = F.avg_pool2d(current_target, 2)
    return loss


class MaskGIT(nn.Module):
    def __init__(self, codebook_size: int, grid_size: int, dimension: int = 384,
                 layers: int = 8, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.config = {"codebook_size": codebook_size, "grid_size": grid_size, "dimension": dimension,
                       "layers": layers, "heads": heads, "dropout": dropout}
        self.codebook_size = codebook_size
        self.mask_id = codebook_size
        self.grid_size = grid_size
        sequence_length = grid_size * grid_size
        self.token_embedding = nn.Embedding(codebook_size + 1, dimension)
        self.position = nn.Parameter(torch.randn(1, sequence_length, dimension) * 0.02)
        block = nn.TransformerEncoderLayer(dimension, heads, dimension * 4, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(block, layers, nn.LayerNorm(dimension))
        self.output = nn.Linear(dimension, codebook_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        flat = tokens.flatten(1)
        x = self.token_embedding(flat) + self.position[:, :flat.shape[1]]
        return self.output(self.transformer(x))


def random_mask(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, height, width = tokens.shape
    count = height * width
    ratios = torch.cos(torch.rand(batch, device=tokens.device) * math.pi / 2)
    amounts = (ratios * count).round().long().clamp(1, count)
    noise = torch.rand(batch, count, device=tokens.device)
    ranks = noise.argsort(1).argsort(1)
    mask = ranks < amounts[:, None]
    masked = tokens.flatten(1).clone()
    masked[mask] = -1
    return masked.view_as(tokens), mask


@torch.inference_mode()
def generate_tokens(model: MaskGIT, batch_size: int, steps: int = 12,
                    temperature: float = 1.0, guidance_noise: float = 1.0,
                    callback=None) -> torch.Tensor:
    device = next(model.parameters()).device
    count = model.grid_size ** 2
    tokens = torch.full((batch_size, count), model.mask_id, dtype=torch.long, device=device)
    for step in range(steps):
        logits = model(tokens.view(batch_size, model.grid_size, model.grid_size)) / max(temperature, 0.05)
        probabilities = logits.softmax(-1)
        sampled = torch.multinomial(probabilities.view(-1, model.codebook_size), 1).view(batch_size, count)
        confidence = probabilities.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        is_masked = tokens.eq(model.mask_id)
        candidates = torch.where(is_masked, sampled, tokens)
        if step == steps - 1:
            tokens = candidates
        else:
            remaining = math.floor(count * math.cos((step + 1) / steps * math.pi / 2))
            score = confidence + guidance_noise * (1 - (step + 1) / steps) * torch.rand_like(confidence)
            score = score.masked_fill(~is_masked, float("inf"))
            remask = score.argsort(1).argsort(1) < remaining
            tokens = candidates.masked_fill(remask, model.mask_id)
        if callback:
            callback(step + 1, steps)
    return tokens.view(batch_size, model.grid_size, model.grid_size)
