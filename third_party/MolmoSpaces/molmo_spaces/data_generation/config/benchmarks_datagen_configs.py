"""
Sepeare file containing imported benchmark tasks.

Why: Some old benchmarks ~18 DEC 2025 were generated with these, and because we pickle the
tasks they need to be accessible in the same location. When no longer using the old
benchmarks this file can be removed.

Moving benchmarks configs to this file caused some complications with imports, so we undid that change.
"""

from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaPickandPlaceDroidBench,  # noqa: F401
    FrankaPickandPlaceDroidMiniBench,  # noqa: F401
    FrankaPickDroidBench,  # noqa: F401
    FrankaPickDroidMiniBench,  # noqa: F401
)
