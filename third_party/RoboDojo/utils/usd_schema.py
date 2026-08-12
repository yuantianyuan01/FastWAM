"""USD schema utilities for setting attributes."""

from typing import Any

import carb


def to_camel_case(snake_str: str) -> str:
    """Convert snake_case to CamelCase (CC format).

    Args:
        snake_str: String in snake_case format

    Returns:
        String in CamelCase format (e.g., friction_combine_mode -> FrictionCombineMode)
    """
    components = snake_str.split("_")
    return "".join(word.capitalize() for word in components)


def safe_set_attribute_on_schema(schema_api, attr_name: str, value: Any):
    """Safely set attribute on USD schema API.

    Args:
        schema_api: USD schema API (e.g., UsdPhysics.MaterialAPI, PhysxSchema.PhysxMaterialAPI)
        attr_name: Attribute name in snake_case (e.g., "friction_combine_mode")
        value: Value to set
    """
    if value is None:
        return

    # Convert to CamelCase for attribute method name
    camel_case_name = to_camel_case(attr_name)
    create_attr_method = getattr(schema_api, f"Create{camel_case_name}Attr", None)

    if create_attr_method is not None:
        create_attr_method().Set(value)
    else:
        # Try direct attribute name if CamelCase doesn't work
        create_attr_method = getattr(schema_api, f"Create{attr_name}Attr", None)
        if create_attr_method is not None:
            create_attr_method().Set(value)
        else:
            carb.log_warn(f"Attribute '{attr_name}' (as '{camel_case_name}') does not exist on schema API.")
