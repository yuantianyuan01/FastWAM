import os

from omegaconf import DictConfig, ListConfig, OmegaConf

from env.global_configs import *


def resolve_path(config_path: str) -> str:
    """Resolve paths from configuration by replacing placeholders with actual directory paths.

    Replaces special path placeholders with their corresponding actual directory paths
    and ensures the result is an absolute path.

    Args:
        config_path: Path string from configuration that may contain placeholders

    Returns:
        Resolved absolute path with placeholders replaced by actual directory paths

    Notes:
        Resolution rules:
        - If path contains `$RoboDojo_ASSETS`, replace with value of the ASSETS_PATH variable
        - Note: `$RoboDojo_*` placeholders are part of the external asset-data contract.
        - If path contains `$NVIDIA_ASSETS`, replace with value of NVIDIA_ASSETS variable
        - If path is absolute (starts with `/`), return as-is without replacement
    """
    # Define placeholder mapping (placeholder -> actual directory path)
    path_mapping = {
        "$RoboDojo_ASSETS": ASSETS_PATH,
        "$Robodojo_ASSETS": ASSETS_PATH,
        "$RoboDojo_CONF": ENV_CONFIG_PATH,
        "$RoboDojo_HOME": ROOT_DIR,
    }

    # 1. Check if path is absolute (no replacement for absolute paths)
    if os.path.isabs(config_path):
        return config_path

    # 2. Replace placeholders in the path
    resolved_path = config_path
    for placeholder, actual_path in path_mapping.items():
        if placeholder in resolved_path:
            resolved_path = resolved_path.replace(placeholder, actual_path)

    # 3. Return the processed path as an absolute path
    return resolved_path


def get_mdl_paths_from_folder(
    folder_path: str,
    recursive: bool = True,
    mdl_paths: list[str] = None,
    skip_keywords: list[str] = None,
) -> list[str]:
    """Retrieve MDL file paths from a folder, optionally searching recursively and filtering by keywords."""
    if mdl_paths is None:
        mdl_paths = []
    skip_keywords = skip_keywords or []

    # Make sure the omni.client extension is enabled
    import omni.kit.app

    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled("omni.client"):
        ext_manager.set_extension_enabled_immediate("omni.client", True)
    import omni.client

    result, entries = omni.client.list(folder_path)
    if result != omni.client.Result.OK:
        print(f"Could list assets in path: {folder_path}")
        return mdl_paths

    for entry in entries:
        if any(keyword.lower() in entry.relative_path.lower() for keyword in skip_keywords):
            continue
        _, ext = os.path.splitext(entry.relative_path)
        if ext in [".mdl"]:
            path_posix = os.path.join(folder_path, entry.relative_path).replace("\\", "/")
            mdl_paths.append(path_posix)
        elif recursive and entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
            sub_folder = os.path.join(folder_path, entry.relative_path).replace("\\", "/")
            get_mdl_paths_from_folder(
                sub_folder,
                recursive=recursive,
                mdl_paths=mdl_paths,
                skip_keywords=skip_keywords,
            )

    return mdl_paths


def get_usd_paths_from_folder(
    folder_path: str,
    recursive: bool = True,
    usd_paths: list[str] = None,
    skip_keywords: list[str] = None,
) -> list[str]:
    """Retrieve USD file paths from a folder, optionally searching recursively and filtering by keywords."""
    if usd_paths is None:
        usd_paths = []
    skip_keywords = skip_keywords or []

    # Make sure the omni.client extension is enabled
    import omni.kit.app

    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled("omni.client"):
        ext_manager.set_extension_enabled_immediate("omni.client", True)
    import omni.client

    result, entries = omni.client.list(folder_path)
    if result != omni.client.Result.OK:
        print(f"Could list assets in path: {folder_path}")
        return usd_paths

    for entry in entries:
        if any(keyword.lower() in entry.relative_path.lower() for keyword in skip_keywords):
            continue
        _, ext = os.path.splitext(entry.relative_path)
        if ext in [".usd", ".usda", ".usdc"]:
            path_posix = os.path.join(folder_path, entry.relative_path).replace("\\", "/")
            usd_paths.append(path_posix)
        elif recursive and entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
            sub_folder = os.path.join(folder_path, entry.relative_path).replace("\\", "/")
            get_usd_paths_from_folder(
                sub_folder,
                recursive=recursive,
                usd_paths=usd_paths,
                skip_keywords=skip_keywords,
            )

    return usd_paths


def deep_resolve_paths(cfg: DictConfig):
    """
    Deeply resolve path placeholders (with "$") in an OmegaConf DictConfig to real accessible paths.

    This function recursively traverses the configuration dictionary and resolves any string values
    containing "$" placeholders to their actual file system paths.

    Args:
        cfg: OmegaConf DictConfig to resolve (e.g., room/object configs like room_top_cfg, cat_spec)
            The configuration is modified in-place.
    """
    for key, value in cfg.items():
        if isinstance(value, DictConfig):
            deep_resolve_paths(value)
        elif isinstance(value, str) and "$" in value:
            OmegaConf.set_struct(cfg, False)
            cfg[key] = resolve_path(value)
            OmegaConf.set_struct(cfg, True)
        elif isinstance(value, ListConfig):
            for i, item in enumerate(value):
                if isinstance(item, DictConfig):
                    deep_resolve_paths(item)
                elif isinstance(item, str) and "$" in item:
                    OmegaConf.set_struct(cfg, False)
                    value[i] = resolve_path(item)
                    OmegaConf.set_struct(cfg, True)
