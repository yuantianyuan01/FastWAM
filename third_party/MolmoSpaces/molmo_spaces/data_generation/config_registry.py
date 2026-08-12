"""
Configuration registry for data generation experiments.

This registry allows multiple config classes to be defined in any file structure,
without strict naming conventions. Simply register your config classes here.
"""

import logging

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig

log = logging.getLogger(__name__)

# Global registry of config classes
_MJT_CONFIG_REGISTRY: dict[str, type[MlSpacesExpConfig]] = {}


def register_config(name: str, strict: bool = True):
    """Decorator to register a config class in the registry.

    Usage:
        @register_config("MyExperimentConfig")
        class MyExperimentConfig(MlSpacesExpConfig):
            ...

    Args:
        name: The name to register this config under (used in command line)
    """

    def decorator(cls: type[MlSpacesExpConfig]):
        if name in _MJT_CONFIG_REGISTRY:
            existing_cls = _MJT_CONFIG_REGISTRY[name]
            existing_cls_id = f"{existing_cls.__module__}.{existing_cls.__name__}"
            new_cls_id = f"{cls.__module__}.{cls.__name__}"
            if strict:
                raise ValueError(
                    f"Config '{name}' already registered as {existing_cls_id}, trying to register as {new_cls_id}"
                )
            log.warning(
                f"Overriding existing config '{name}'. Was {existing_cls_id}, now {new_cls_id}"
            )
        _MJT_CONFIG_REGISTRY[name] = cls
        return cls

    return decorator


def get_config_class(name: str) -> type[MlSpacesExpConfig]:
    """Get a config class by name from the registry.

    Args:
        name: The name of the config class

    Returns:
        The config class

    Raises:
        ValueError: If the config name is not found
    """
    if name not in _MJT_CONFIG_REGISTRY:
        available = list(_MJT_CONFIG_REGISTRY.keys())
        raise ValueError(f"Config '{name}' not found. Available configs: {available}")
    return _MJT_CONFIG_REGISTRY[name]


def list_available_configs() -> list[str]:
    """List all available config names in the registry."""
    return list(_MJT_CONFIG_REGISTRY.keys())


def get_registry_size() -> int:
    """Get the number of registered configs."""
    return len(_MJT_CONFIG_REGISTRY)
