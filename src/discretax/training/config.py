"""Experiment configuration loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _expect_keys(data: dict[str, Any], allowed: set[str], *, context: str) -> None:
    """Raise when a configuration mapping contains unexpected keys."""
    unknown_keys = sorted(set(data) - allowed)
    if unknown_keys:
        raise ValueError(f"Unexpected keys in {context}: {unknown_keys}")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries without mutating the inputs."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML file and validate that it contains a mapping."""
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in config file {path}, got {type(loaded).__name__}")
    return loaded


def _resolve_default_path(reference: str, current_path: Path) -> Path:
    """Resolve a defaults entry relative to the current config file."""
    default_path = Path(reference)
    if not default_path.suffix:
        default_path = default_path.with_suffix(".yaml")
    if not default_path.is_absolute():
        default_path = (current_path.parent / default_path).resolve()
    if not default_path.exists():
        raise FileNotFoundError(f"Default config '{reference}' resolved to missing path {default_path}")
    return default_path


def _resolve_defaults(path: Path, visited: set[Path]) -> dict[str, Any]:
    """Resolve a config file and its recursive defaults chain."""
    path = path.resolve()
    if path in visited:
        raise ValueError(f"Detected recursive config defaults while resolving {path}")
    visited.add(path)
    raw_config = _load_yaml_mapping(path)
    defaults = raw_config.pop("defaults", [])
    if not isinstance(defaults, list):
        raise ValueError(f"'defaults' in {path} must be a list of config paths")

    merged: dict[str, Any] = {}
    for reference in defaults:
        if not isinstance(reference, str):
            raise ValueError(f"Defaults entries in {path} must be strings, got {reference!r}")
        default_path = _resolve_default_path(reference, path)
        merged = _deep_merge(merged, _resolve_defaults(default_path, visited))

    visited.remove(path)
    return _deep_merge(merged, raw_config)


def _parse_override_value(raw_value: str) -> Any:
    """Parse a dotted override value using YAML scalar semantics."""
    return yaml.safe_load(raw_value)


def _set_dotted_key(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested configuration key addressed by dot notation."""
    keys = dotted_key.split(".")
    if not keys or any(not key for key in keys):
        raise ValueError(f"Invalid override key: {dotted_key!r}")

    current = config
    for key in keys[:-1]:
        next_value = current.get(key)
        if next_value is None:
            next_value = {}
            current[key] = next_value
        if not isinstance(next_value, dict):
            raise ValueError(
                f"Cannot apply override '{dotted_key}': '{key}' is not a mapping in the config"
            )
        current = next_value
    current[keys[-1]] = value


def _apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dotted-key overrides to a config dictionary."""
    updated = dict(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override '{override}' must have the form key=value")
        dotted_key, raw_value = override.split("=", 1)
        _set_dotted_key(updated, dotted_key, _parse_override_value(raw_value))
    return updated


@dataclass(slots=True)
class PathsConfig:
    """Filesystem locations for experiment assets."""

    data_root: str = "data"
    output_root: str = "outputs"


@dataclass(slots=True)
class DataloaderConfig:
    """Batching and shuffling configuration."""

    batch_size: int = 64
    eval_batch_size: int | None = None
    shuffle_train: bool = True
    drop_last_train: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("loader.batch_size must be positive")
        if self.eval_batch_size is not None and self.eval_batch_size <= 0:
            raise ValueError("loader.eval_batch_size must be positive when provided")

    @property
    def resolved_eval_batch_size(self) -> int:
        """Return the evaluation batch size, defaulting to the train batch size."""
        return self.eval_batch_size or self.batch_size


@dataclass(slots=True)
class DatasetConfig:
    """Dataset selection and preprocessing configuration."""

    kind: str
    name: str | None = None
    root: str | None = None
    download: bool = False
    validation_split: float = 0.1
    sequence_layout: str = "rows"
    normalize: bool = True
    seed: int = 0
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.validation_split < 1.0:
            raise ValueError("dataset.validation_split must be in the range [0.0, 1.0)")
        if not self.kind:
            raise ValueError("dataset.kind must be set")

    @property
    def resolved_name(self) -> str:
        """Return the display name for the dataset."""
        return self.name or self.kind


@dataclass(slots=True)
class ComponentConfig:
    """Config for a model component resolved from a target string."""

    target: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.target:
            raise ValueError("component.target must be set")

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, context: str) -> ComponentConfig:
        """Build a component config from a mapping."""
        if "target" not in data:
            raise ValueError(f"{context}.target must be set")

        kwargs = dict(data.get("kwargs", {}))
        if not isinstance(kwargs, dict):
            raise ValueError(f"{context}.kwargs must be a mapping")

        for key, value in data.items():
            if key not in {"target", "kwargs"}:
                kwargs[key] = value

        return cls(target=str(data["target"]), kwargs=kwargs)


@dataclass(slots=True)
class ModelConfig:
    """Configuration for the encoder/backbone/head model stack."""

    name: str
    hidden_dim: int
    encoder: ComponentConfig
    backbone: ComponentConfig
    head: ComponentConfig

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("model.hidden_dim must be positive")
        if not self.name:
            raise ValueError("model.name must be set")


@dataclass(slots=True)
class ScheduleConfig:
    """Learning-rate schedule configuration."""

    name: str = "constant"
    warmup_steps: int = 0
    decay_steps: int | None = None
    end_value: float = 0.0

    def __post_init__(self) -> None:
        supported = {"constant", "cosine_decay", "warmup_cosine_decay"}
        if self.name not in supported:
            raise ValueError(f"optimizer.schedule.name must be one of {sorted(supported)}")
        if self.warmup_steps < 0:
            raise ValueError("optimizer.schedule.warmup_steps must be non-negative")
        if self.decay_steps is not None and self.decay_steps <= 0:
            raise ValueError("optimizer.schedule.decay_steps must be positive when provided")


@dataclass(slots=True)
class OptimizerConfig:
    """Optimizer and regularisation configuration."""

    name: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    grad_clip_norm: float | None = None
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)

    def __post_init__(self) -> None:
        supported = {"adam", "adamw", "sgd"}
        if self.name not in supported:
            raise ValueError(f"optimizer.name must be one of {sorted(supported)}")
        if self.learning_rate <= 0:
            raise ValueError("optimizer.learning_rate must be positive")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("optimizer.grad_clip_norm must be positive when provided")


@dataclass(slots=True)
class TrainerConfig:
    """Runtime training configuration."""

    seed: int = 0
    num_epochs: int = 1
    max_steps: int | None = None
    log_every_steps: int = 10
    eval_every_steps: int = 100
    checkpoint_every_steps: int = 100
    jit: bool = True

    def __post_init__(self) -> None:
        if self.num_epochs <= 0:
            raise ValueError("trainer.num_epochs must be positive")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("trainer.max_steps must be positive when provided")
        if self.log_every_steps <= 0:
            raise ValueError("trainer.log_every_steps must be positive")
        if self.eval_every_steps <= 0:
            raise ValueError("trainer.eval_every_steps must be positive")
        if self.checkpoint_every_steps <= 0:
            raise ValueError("trainer.checkpoint_every_steps must be positive")


@dataclass(slots=True)
class CheckpointConfig:
    """Checkpoint policy for experiments."""

    enabled: bool = True
    save_best: bool = True
    monitor: str = "val_loss"
    mode: str = "min"

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("checkpoint.mode must be either 'min' or 'max'")


@dataclass(slots=True)
class WandbConfig:
    """Weights & Biases logging configuration."""

    enabled: bool = False
    project: str | None = None
    entity: str | None = None
    group: str | None = None
    job_type: str | None = None
    mode: str | None = None
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str
    description: str | None
    tags: list[str]
    paths: PathsConfig
    loader: DataloaderConfig
    dataset: DatasetConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig
    checkpoint: CheckpointConfig
    wandb: WandbConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Build and validate an experiment config from a dictionary."""
        allowed_keys = {
            "name",
            "description",
            "tags",
            "paths",
            "loader",
            "dataset",
            "model",
            "optimizer",
            "trainer",
            "checkpoint",
            "wandb",
        }
        _expect_keys(data, allowed_keys, context="experiment config")

        paths = PathsConfig(**data.get("paths", {}))
        loader = DataloaderConfig(**data.get("loader", {}))
        dataset = DatasetConfig(**data["dataset"])

        model_data = data["model"]
        _expect_keys(
            model_data,
            {"name", "hidden_dim", "encoder", "backbone", "head"},
            context="model config",
        )
        model = ModelConfig(
            name=str(model_data["name"]),
            hidden_dim=int(model_data["hidden_dim"]),
            encoder=ComponentConfig.from_dict(model_data["encoder"], context="model.encoder"),
            backbone=ComponentConfig.from_dict(model_data["backbone"], context="model.backbone"),
            head=ComponentConfig.from_dict(model_data["head"], context="model.head"),
        )

        optimizer_data = dict(data.get("optimizer", {}))
        schedule = ScheduleConfig(**optimizer_data.pop("schedule", {}))
        optimizer = OptimizerConfig(schedule=schedule, **optimizer_data)
        trainer = TrainerConfig(**data.get("trainer", {}))
        checkpoint = CheckpointConfig(**data.get("checkpoint", {}))
        wandb = WandbConfig(**data.get("wandb", {}))

        tags = data.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError("tags must be a list of strings")

        name = str(data["name"])
        if not name:
            raise ValueError("name must be set")

        return cls(
            name=name,
            description=data.get("description"),
            tags=tags,
            paths=paths,
            loader=loader,
            dataset=dataset,
            model=model,
            optimizer=optimizer,
            trainer=trainer,
            checkpoint=checkpoint,
            wandb=wandb,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the experiment config back to a plain dictionary."""
        return asdict(self)


def load_experiment_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load and validate an experiment config file."""
    raw_config = _resolve_defaults(Path(path), visited=set())
    if overrides:
        raw_config = _apply_overrides(raw_config, overrides)
    return ExperimentConfig.from_dict(raw_config)


def flatten_config(config: ExperimentConfig | dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested config into dotted keys for logging backends."""
    if isinstance(config, ExperimentConfig):
        raw_config = config.to_dict()
    else:
        raw_config = config

    flattened: dict[str, Any] = {}

    def _flatten(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, nested_value in value.items():
                nested_prefix = f"{prefix}.{key}" if prefix else key
                _flatten(nested_prefix, nested_value)
        else:
            flattened[prefix] = value

    _flatten("", raw_config)
    return flattened
