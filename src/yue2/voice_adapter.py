"""Opt-in reference-conditioned NAR experiment; never imported by the base pipeline."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .nar import CachedNAR, attention, song_chunks


class VoiceAdapter(nn.Module):
    """Zero-initialized, low-rank FiLM residuals on selected NAR layers."""

    def __init__(self, hidden_size, num_layers, reference_dim, rank=64, start_layer=0):
        super().__init__()
        if min(hidden_size, num_layers, reference_dim, rank) < 1:
            raise ValueError("Adapter dimensions must be positive")
        if not 0 <= start_layer < num_layers:
            raise ValueError("start_layer is outside the backbone")
        self.spec = dict(hidden_size=hidden_size, num_layers=num_layers,
                         reference_dim=reference_dim, rank=rank, start_layer=start_layer)
        self.reference = nn.Sequential(nn.Linear(reference_dim, rank), nn.SiLU())
        self.modulations = nn.ModuleDict({
            str(index): nn.Linear(rank, 2 * hidden_size)
            for index in range(start_layer, num_layers)
        })
        for projection in self.modulations.values():
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def encode_reference(self, reference):
        reference = reference.to(device=self.reference[0].weight.device, dtype=torch.float32)
        if reference.shape != (self.spec["reference_dim"],):
            raise ValueError("Expected one reference embedding [reference_dim]")
        if not torch.isfinite(reference).all() or reference.norm() < 1e-8:
            raise ValueError("Reference embedding must be finite and nonzero")
        return self.reference(F.normalize(reference, dim=0))

    def forward(self, hidden, encoded_reference, layer_index, strength=1.0):
        key = str(layer_index)
        if key not in self.modulations or strength == 0:
            return hidden
        scale, shift = self.modulations[key](encoded_reference).chunk(2)
        normalized = F.layer_norm(hidden.float(), (hidden.shape[-1],))
        residual = normalized * scale + shift
        return hidden + (strength * residual).to(hidden.dtype)


class VoiceChunkFlow(CachedNAR):
    """Differentiable counterpart of CachedNAR with frozen AR prefill."""

    def __init__(self, model, chunk, adapter, reference, *, strength=1.0,
                 checkpoint_layers=False):
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("strength must be finite and nonnegative")
        expected = (model.config.hidden_size, model.config.num_hidden_layers)
        if (adapter.spec["hidden_size"], adapter.spec["num_layers"]) != expected:
            raise ValueError("Adapter and backbone dimensions differ")
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Freeze all base model parameters before using VoiceChunkFlow")
        if model.training:
            raise ValueError("The frozen base model must remain in eval mode")
        self.adapter = adapter
        self.reference = reference
        self.strength = strength
        self.checkpoint_layers = checkpoint_layers
        super().__init__(model, chunk)

    @torch.no_grad()
    def _prefill(self):
        backbone = self.model.model
        ids = torch.tensor([self.chunk.ar_tokens], dtype=torch.long, device=self.device)
        positions = torch.arange(self.ar_length, device=self.device)[None]
        cos, sin = backbone.rotary_emb(positions)
        hidden = backbone.embed_tokens(ids)
        for layer in backbone.layers:
            query, key, value = layer.self_attn.project_qkv(layer.input_layernorm(hidden), cos, sin)
            self.cache.append((key[0, :self.visible_length].clone(),
                               value[0, :self.visible_length].clone()))
            attended = self._attention(query[0], key[0], value[0], causal=True)
            hidden = hidden + layer.self_attn.o_proj(attended.flatten(1)[None])
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))

    def _layer(self, hidden, reference, layer_index):
        layer = self.model.model.layers[layer_index]
        ar_key, ar_value = self.cache[layer_index]
        query, key, value = layer.nar_self_attn.project_qkv(
            layer.nar_input_layernorm(hidden), self.cos, self.sin)
        attended = attention(query[0], torch.cat((ar_key, key[0])),
                             torch.cat((ar_value, value[0])))
        hidden = hidden + layer.nar_self_attn.o_proj(attended.flatten(1)[None])
        hidden = self.adapter(hidden, reference, layer_index, self.strength)
        return hidden + layer.nar_mlp(layer.nar_pre_mlp_layernorm(hidden))

    def velocity(self, state, raw_t):
        if tuple(state.shape) != tuple(self.chunk.noise.shape):
            raise ValueError("ODE state shape changed")
        model = self.model
        state = state.to(device=self.device, dtype=self.dtype)
        shifted = model._shift_t_value(raw_t, self.device, self.dtype)
        hidden = model.vae2llm(F.pad(state, (0, 0, 1, 1))[None])
        hidden = hidden + model.time_embedder(shifted.expand(self.nar_length))[None]
        hidden = hidden + self.pos_emb
        reference = self.adapter.encode_reference(self.reference)
        for layer_index in range(len(model.model.layers)):
            if self.checkpoint_layers and torch.is_grad_enabled():
                def run_layer(current, condition, index=layer_index):
                    return self._layer(current, condition, index)
                hidden = checkpoint(run_layer, hidden, reference, use_reentrant=False)
            else:
                hidden = self._layer(hidden, reference, layer_index)
        return model.llm2vae(model.model.norm(hidden))[0, 1:-1]


def flow_loss(engine, target, *, time=None, noise=None):
    """Train clean at t=0, noise at t=1, matching the existing reverse solver."""
    target = target.to(device=engine.device, dtype=torch.float32)
    if target.shape != engine.chunk.noise.shape or not torch.isfinite(target).all():
        raise ValueError("Expected finite target latents [frames,64] matching semantic length")
    if time is None:
        time = torch.rand(()).item() * (1 - 2e-5) + 1e-5
    if not 0 < time < 1:
        raise ValueError("time must be strictly between zero and one")
    noise = torch.randn_like(target) if noise is None else noise.to(target)
    if noise.shape != target.shape or not torch.isfinite(noise).all():
        raise ValueError("Noise must match target latents and be finite")
    state = (1 - time) * target + time * noise
    predicted = engine.velocity(state, math.log(time / (1 - time)))
    return F.mse_loss(predicted.float(), noise - target)


@torch.inference_mode()
def synthesize_voice(model, prefix, codec, seed, adapter, reference, *, steps=32,
                     strength=1.0, context=24576):
    """Separate synthesis API; the original pipeline and its weights are untouched."""
    output = []
    for chunk in song_chunks(prefix, codec, seed, context=context):
        engine = VoiceChunkFlow(model, chunk, adapter, reference, strength=strength)
        try:
            output.append(engine.solve(steps=steps))
        finally:
            engine.close()
    return torch.cat(output)
