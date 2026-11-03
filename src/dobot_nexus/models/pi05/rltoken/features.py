from __future__ import annotations

import json

import torch

from dobot_nexus.algorithms.rltoken.artifacts import file_digest
from dobot_nexus.algorithms.rltoken.identity import content_digest
from dobot_nexus.core.config import to_plain_dict
from dobot_nexus.core.contracts import RLFeatureBatch, RLFeatureSpec


def feature_spec(model, *, weights=None):
    pipeline = model.adapter.pipeline
    if pipeline.norm_stats_path is None:
        raise ValueError("RL feature export requires resolved normalization statistics.")
    settings = model.root_config.model.rltoken
    codec = settings["action_codec"]
    identity = content_digest(
        {
            "schema": "pi05_rlt_features_v1",
            "weights": {
                key: value
                for key, value in (weights if weights is not None else model.state_dict()).items()
                if key.startswith(("core_model.", "token_encoder."))
            },
            "core": model.pi05_config.core_kwargs(),
            "token": to_plain_dict(settings["token"]),
            "codec": to_plain_dict(codec),
            "normalization": pipeline.normalization,
            "stats": json.loads(pipeline.norm_stats_path.read_text()),
            "tokenizer": file_digest(pipeline.tokenizer_path),
            "operator_numerics": model.pi05_config.operator_numerics,
            "image_size": model.pi05_config.image_size,
            "camera_names": model.pi05_config.camera_names,
            **({"camera_mapping": model.pi05_config.camera_mapping}
               if model.pi05_config.camera_mapping != {
                   "base": "base_0_rgb", "left_wrist": "left_wrist_0_rgb",
                   "right_wrist": "right_wrist_0_rgb",
               } else {}),
            "delta_mask": pipeline.delta_mask.tolist(),
            "control_dt": settings["control_dt"],
            "inference_profile": ("native_bf16_parameters_v1" if model.pi05_config.parameter_precision == "bf16"
                                  else "autocast_fp32_parameters_v1"),
        }
    )
    return RLFeatureSpec(
        model.token_config.width,
        model.pi05_config.model_action_dim,
        model.pi05_config.action_dim,
        model.pi05_config.action_horizon,
        identity,
        model.adapter.codec_id,
        f"pi05_{pipeline.normalization}_padded_v1",
        codec["units"],
        settings["control_dt"],
        tuple(codec["low"]),
        tuple(codec["high"]),
    )


@torch.no_grad()
def extract_features(model, observation, options):
    if model.training:
        raise RuntimeError("Frozen RL features require eval mode.")
    encoded = model.adapter.encode_observation(observation)
    device = next(model.core_model.parameters()).device
    encoded = model.adapter.move_encoded(encoded, device)
    if encoded.model_kwargs:
        raise ValueError("RLToken v1 requires stateless Pi0.5 features.")
    steps = options.num_steps if options.num_steps is not None else model.pi05_config.num_steps
    if type(steps) is not int or steps <= 0:
        raise ValueError("Reference num_steps must be positive.")
    seed = options.seed if options.seed is not None else (0 if options.deterministic else None)
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    noise = None
    if options.noise is not None:
        noise = torch.as_tensor(options.noise, device=device, dtype=torch.float32)
        expected = (
            observation.batch_size,
            model.pi05_config.action_horizon,
            model.pi05_config.model_action_dim,
        )
        if noise.shape != expected or not torch.isfinite(noise).all():
            raise ValueError(f"Reference noise must be finite with shape {expected}.")
    captured = {}

    def prefix_once(prepared):
        prefix, mask, kv = model.core_model.build_prefix_cache(prepared)
        selected, valid = select_token_prefix(model, prefix, mask, prepared)
        with torch.autocast(device.type, dtype=model.pi05_config.torch_dtype,
                            enabled=model.pi05_config.torch_dtype != torch.float32):
            captured["latent"] = model.token_encoder(selected, valid)
        return prefix, mask, kv

    dtype = model.pi05_config.torch_dtype
    with torch.autocast(device.type, dtype=dtype,
                        enabled=dtype != torch.float32 and model.pi05_config.parameter_precision == "fp32"):
        actions = model.core_model.sample_actions(
            encoded.observation, num_steps=steps, noise=noise, rng=generator, _prefix_fn=prefix_once
        )
    raw = model.adapter.decode_vla_actions(
        actions.detach().cpu().numpy(), aux={"raw_state": encoded.raw_state}
    ).actions
    context = {
        "raw_state": encoded.raw_state.tolist(),
        "observation_id": content_digest(
            {"state": observation.state, "images": observation.images, "prompt": observation.prompt}
        ),
    }
    commands = model.adapter.encode_executed_actions(
        raw, aux={**context, "executed_steps": [raw.shape[1]] * raw.shape[0]}
    )
    if not commands.encodable_mask.all():
        raise ValueError(
            "VLA reference is outside the configured RL command bounds; inspect the codec/recipe."
        )
    spec = model.rl_feature_spec
    packet = RLFeatureBatch(
        captured["latent"].detach(),
        encoded.observation.state.detach(),
        commands.actions,
        commands.executed_mask,
        spec.feature_id,
        context,
        [
            {"raw_state": [state], "observation_id": context["observation_id"]}
            for state in context["raw_state"]
        ],
    )
    return packet.cpu_copy().validate(spec)


def select_token_prefix(model, prefix, valid, observation):
    if model.token_config.selection == "full":
        return prefix, valid
    # Pi0.5 embeds cameras first. Derive positions from its own vision operator.
    patch_h, patch_w = model.core_model.img.patch_size
    count = sum((image.shape[-3] // patch_h) * (image.shape[-2] // patch_w)
                for image in observation.images.values())
    if count <= 0 or count > prefix.shape[1]:
        raise ValueError("Pi0.5 image token layout disagrees with prefix shape.")
    return prefix[:, :count], valid[:, :count]
