"""Token-only job using the public model/data-adapter contracts."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from dobot_nexus.core.config import clone_config
from dobot_nexus.core.contracts import TrainResult
from dobot_nexus.core.import_utils import import_from_string
from dobot_nexus.training.resume import capture_rng_state, restore_rng_state
from .artifacts import write_model_bundle, feature_metadata
from .checkpoint import _fsync
from .identity import content_digest


class TokenPretrainTrainer:
    def __init__(self, config, experiment):
        self.config, self.experiment = config, experiment
        settings = config.algorithm.token
        train, validation = set(settings.train_episodes), set(settings.validation_episodes)
        if not train or not validation or train & validation:
            raise ValueError("Token train/validation must be nonempty disjoint episode sets.")
        if config.model.rltoken.get("training_mode") != "token_only":
            raise ValueError("TokenPretrainTrainer requires model.rltoken.training_mode=token_only.")
        if not config.data.get("norm_stats_path"):
            raise ValueError("Token training requires the frozen provider's existing normalization asset.")
        if min(config.train.max_steps, config.train.save_every, config.data.batch_size) <= 0:
            raise ValueError("Token budgets and batch size must be positive.")
        for key in ("min_zero_ratio", "min_shuffle_ratio"):
            if settings.get(key, 0) <= 1:
                raise ValueError(f"Declare token.{key} > 1 for the dependence gate.")

    def run(self):
        cfg, root = self.config, Path(self.experiment.run_dir) / "token"
        root.mkdir(parents=True, exist_ok=True)
        settings = cfg.algorithm.token
        device = torch.device(cfg.runtime.get("device", "cpu"))
        precision = cfg.runtime.get("precision", cfg.model.get("precision", "fp32"))
        if precision not in {"fp32", "bf16"}:
            raise ValueError("TokenPretrainTrainer supports fp32 or bf16 precision.")
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        torch.manual_seed(int(cfg.exp.get("seed", 0)))
        model = import_from_string(cfg.model.target).from_config(cfg, mode="train").to(device)
        train_cfg, val_cfg = clone_config(cfg), clone_config(cfg)
        train_cfg.data.episode_ids = list(settings.train_episodes)
        val_cfg.data.episode_ids = list(settings.validation_episodes)
        train_cfg.data.num_workers = val_cfg.data.num_workers = 0
        loader_target = settings.get("data_loader")
        build_loader = (lambda conf: import_from_string(loader_target)(model, conf)) if loader_target else model.build_train_dataloader
        train_loader = build_loader(train_cfg)
        validation_loader = build_loader(val_cfg)
        # Final model config describes training data; normalization/prompt assets are shared.
        if not loader_target:
            model.build_train_dataloader(train_cfg)
        fingerprint = content_digest({"model": dict(cfg.model), "data": dict(cfg.data),
                                      "token": dict(settings), "optim": dict(cfg.get("optim", {})),
                                      "precision": precision, "seed": cfg.exp.get("seed", 0)})
        optimizer = torch.optim.AdamW(model.get_param_groups(cfg.get("optim", {})),
            lr=cfg.get("optim", {}).get("lr", 1e-4), weight_decay=cfg.get("optim", {}).get("weight_decay", 0.))
        generator = torch.Generator().manual_seed(int(cfg.exp.get("seed", 0)))
        start, best, checkpoint = 0, float("inf"), ""
        resume = cfg.train.get("resume_from")
        if resume:
            state = torch.load(resume, map_location="cpu", weights_only=False)
            if state["fingerprint"] != fingerprint:
                raise ValueError("Token resume configuration mismatch.")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            generator.set_state(state["sampler"].cpu())
            restore_rng_state(state["rng"])
            start, best = state["step"], state["best"]
        metrics = {}
        for step in range(start, cfg.train.max_steps):
            indices = torch.randint(len(train_loader.dataset), (cfg.data.batch_size,), generator=generator)
            raw = train_loader.collate_fn([train_loader.dataset[int(i)] for i in indices])
            model.train()
            batch = model.adapter.encode_train_batch(raw, device)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                output = model.training_step(batch)
            optimizer.zero_grad(set_to_none=True)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                          1., error_if_nonfinite=True)
            optimizer.step()
            if (step + 1) % cfg.train.get("log_every", 20) == 0:
                self.experiment.logger.info("Token step=%d train_loss=%g", step + 1, float(output.loss.detach()))
            if (step + 1) % cfg.train.save_every and step + 1 != cfg.train.max_steps:
                continue
            model.eval()
            values, count = {}, 0
            with torch.no_grad():
                for offset in range(0, len(validation_loader.dataset), cfg.data.batch_size):
                    rows = [validation_loader.dataset[i] for i in range(offset,
                            min(offset + cfg.data.batch_size, len(validation_loader.dataset)))]
                    raw = validation_loader.collate_fn(rows)
                    with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                        result = model.training_step(model.adapter.encode_train_batch(raw, device))
                    for key, value in result.metrics.items():
                        values[key] = values.get(key, 0.) + float(value) * len(rows)
                    count += len(rows)
            if not count:
                raise ValueError("Token validation is empty.")
            metrics = {key: value / count for key, value in values.items()}
            accepted = (metrics["zero_ratio"] >= settings.min_zero_ratio and
                        metrics["shuffle_ratio"] >= settings.min_shuffle_ratio)
            artifact = root / f"step-{step+1:09d}"
            write_model_bundle(artifact, model, train_cfg)
            _, spec, _ = feature_metadata(artifact)
            (artifact / "token_validation.json").write_text(json.dumps({**metrics, "feature_id": spec.feature_id,
                "accepted": accepted, "train_episodes": list(settings.train_episodes),
                "validation_episodes": list(settings.validation_episodes)}, indent=2))
            if metrics["token_loss"] < best:
                best = metrics["token_loss"]
                (root / "best").write_text(artifact.name)
            (root / "last").write_text(artifact.name)
            temporary = root / ".resume.pt"
            torch.save(dict(fingerprint=fingerprint, model=model.state_dict(), optimizer=optimizer.state_dict(),
                sampler=generator.get_state(), rng=capture_rng_state(), step=step+1, best=best), temporary)
            _fsync(temporary)
            temporary.replace(root / "resume.pt")
            checkpoint = str(root / "resume.pt")
            self.experiment.logger.info("Token step=%d validation=%s accepted=%s", step+1, metrics, accepted)
        return TrainResult(str(self.experiment.run_dir), checkpoint, cfg.train.max_steps, metrics)
