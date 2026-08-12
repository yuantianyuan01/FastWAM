"""Simple configuration management using Pydantic.
This module provides a base configuration class that can be extended to create specific configurations.
It uses Pydantic for data validation and can return dicts or jsons or save or load jsons from files.
"""

from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    """
    Base configuration class that can be extended for specific configurations.
    Provides methods to convert to dict, json, and to save/load from files.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # allow arbitrary types in the config
        # frozen=True,  # (Optional) Make the config immutable
    )

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return self.model_dump()

    def to_json(self) -> str:
        # TODO(max): this can cause errors, try printing out the config to see missmatches w/ print(self)
        return self.model_dump_json(warnings="error")

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create a configuration instance from a dictionary."""
        return cls.model_validate(data)

    @classmethod
    def load_from_json(cls, file_path: str) -> "Config":
        """Load the configuration from a JSON file."""
        with open(file_path, "r") as f:
            data = f.read()
        return cls.model_validate_json(data)

    def save_to_json(self, file_path: str) -> None:
        """Save the configuration to a JSON file."""
        with open(file_path, "w") as f:
            f.write(self.to_json())
