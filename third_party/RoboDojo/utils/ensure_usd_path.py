from env.global_configs import USD_PATH


def ensure_usd_path(obj):
    if USD_PATH is None:
        return
    new_prefix = USD_PATH
    CDN_PREFIX = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0"

    if hasattr(obj, "__dict__"):
        for key, value in obj.__dict__.items():
            if isinstance(value, str) and value.startswith(CDN_PREFIX):
                suffix = value.split("Assets/Isaac/5.0", 1)[1]
                new_path = f"file://{new_prefix}/Assets/Isaac/5.0{suffix}"
                setattr(obj, key, new_path)
            else:
                ensure_usd_path(value)

    elif isinstance(obj, (list, tuple)):
        for item in obj:
            ensure_usd_path(item)

    elif isinstance(obj, dict):
        for value in obj.values():
            ensure_usd_path(value)
