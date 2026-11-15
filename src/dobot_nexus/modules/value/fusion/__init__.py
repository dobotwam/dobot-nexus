"""Optional Fusion value estimator; model and tokenizer dependencies are lazy."""


def require_value_dependencies():
    from importlib.metadata import PackageNotFoundError, version
    from pathlib import Path

    try:
        installed = version("transformers")
    except PackageNotFoundError:
        installed = "missing"
    if installed != "4.53.2":
        requirements = Path(__file__).with_name("requirements.txt")
        raise ImportError(
            f"Dobot Fusion requires transformers==4.53.2 (found {installed}). "
            f"Install the optional dependencies listed in {requirements} "
            "in the environment running Value."
        )


def build_value_model(config):
    config.validate_assets()
    require_value_dependencies()
    from .modeling_fusion import DobotValueFusion

    return DobotValueFusion(config)
