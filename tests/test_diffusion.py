"""Diffusion estimates: components from repo files, params from a safetensors header, latent tokens, stages."""

import json
import struct

import httpx

from kb.research import diffusion as d

FILES = [{"type": "file", "path": p, "lfs": {"size": int(gb * 2**30)}} for p, gb in [
    ("diffusion_models/h3_ref2va_pruned_bf16.safetensors", 37.5), ("diffusion_models/h3_ref2va_pruned_int8.safetensors", 19.5),
    ("diffusion_models/h3_fl2va_bf16.safetensors", 61.7), ("text_encoders/qwen3vl_32b_bf16.safetensors", 48.0),
    ("text_encoders/qwen3vl_32b_int8.safetensors", 25.3), ("vae/h3_video_vae_fp16.safetensors", 4.9),
    ("loras/turbo_8step_bf16.safetensors", 1.8)]]
T_CFG = {"num_attention_heads": 56, "attention_head_dim": 128, "hidden_size": 5376, "num_layers": 50,
         "num_refiner_layers": 2, "ffn_dim": 14336, "patch_size": [1, 2, 2]}
V_CFG = {"spatial_downsample_factors": [2, 2, 2, 2, 1, 1], "temporal_downsample_factors": [1, 2, 2, 1, 1, 1]}
HEADER = json.dumps({"__metadata__": {}, "w": {"dtype": "I8", "shape": [20_000, 1_000_000]},
                     "n": {"dtype": "BF16", "shape": [100_000_000]}}).encode()


class Fake:
    def get(self, url, params=None, headers=None):
        if "/tree/main" in url:
            return httpx.Response(200, json=[{"type": "file", "path": "FL2VA/transformer/config.json"},
                                             {"type": "file", "path": "FL2VA/vae/config.json"}])
        if url.endswith("transformer/config.json"):
            return httpx.Response(200, json=T_CFG)
        if url.endswith("vae/config.json"):
            return httpx.Response(200, json=V_CFG)
        rng = (headers or {}).get("Range", "")
        if rng == "bytes=0-7":
            return httpx.Response(206, content=struct.pack("<Q", len(HEADER)))
        return httpx.Response(206, content=HEADER)

    def close(self):
        pass


def test_components_and_roles():
    comps = d.components(None, "x", FILES)
    assert [v["precision"] for v in comps["denoiser"]] == ["int8", "bf16", "bf16"] and "loras" not in str(comps)
    assert d._pick(comps["denoiser"], "int8", "ref2va_pruned")["name"].endswith("pruned_int8.safetensors")
    diffusers = d.components(None, "x", [{"type": "file", "path": "vae/diffusion_pytorch_model.safetensors", "size": 1},
                                         {"type": "file", "path": "transformer/diffusion_pytorch_model-00001-of-00002.safetensors", "size": 1},
                                         {"type": "file", "path": "transformer/diffusion_pytorch_model-00002-of-00002.safetensors", "size": 1}])
    assert diffusers["vae"][0]["name"] == "vae" and len(diffusers["denoiser"][0]["files"]) == 2


def test_video_estimate_tokens_stages_and_offload(monkeypatch):
    monkeypatch.setattr(d, "interactive_client", lambda *a, **k: Fake())
    meta = {"cardData": {"base_model": ["MiniMaxAI/MiniMax-H3"]}}
    r = d.estimate_diffusion("Comfy-Org/MiniMax-H3", meta, FILES, precision="int8", variant="ref2va_pruned",
                             width=1344, height=768, frames=124, steps=10, gpu={"name": "RTX A6000", "vram_gb": 48},
                             gpu_tflops=155, ram_gb=64)
    assert r["denoiser"]["params_b"] == 20.1 and r["text_encoder"]["file"].endswith("int8.safetensors")
    assert r["latent_tokens"] == 31 * 24 * 42                        # (124-1)//4+1 frames, 768/16/2, 1344/16/2
    assert r["fits"] and 5 < r["total_minutes"] < 8
    small = d.estimate_diffusion("Comfy-Org/MiniMax-H3", meta, FILES, precision="int8", variant="ref2va_pruned",
                                 width=1344, height=768, frames=124, gpu={"name": "3060 Ti", "vram_gb": 8},
                                 gpu_tflops=32, ram_gb=30)
    assert not small["fits"] and "every step" in small["offload"]["denoise"] and small["ram_ok"] is False
