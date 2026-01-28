"""
Train/test split definitions for proper SC-SINDy evaluation.

This module defines which systems are used for training vs testing the structure
network. The key principle is that TEST systems should never be seen during training
to ensure fair evaluation of generalization.

Note: Lorenz system is intentionally held out as the canonical SINDy benchmark.
"""

from typing import Dict, List, Tuple, Type

from ..systems import (
    ChenSystem,
    CoupledBrusselator,
    DampedHarmonicOscillator,
    DuffingOscillator,
    # Base
    DynamicalSystem,
    # 3D Chaotic
    Lorenz,
    # 2D Biological
    LotkaVolterra,
    Rossler,
    SelkovGlycolysis,
    # 3D Biological
    SIRModel,
    # 2D Oscillators
    VanDerPol,
)

# =============================================================================
# 2D System Splits
# =============================================================================

TRAIN_SYSTEMS_2D: List[Type[DynamicalSystem]] = [
    VanDerPol,
    DuffingOscillator,
    LotkaVolterra,
    SelkovGlycolysis,
]

TEST_SYSTEMS_2D: List[Type[DynamicalSystem]] = [
    DampedHarmonicOscillator,
    CoupledBrusselator,
]

# =============================================================================
# 3D System Splits (Deferred - insufficient systems for meaningful split)
# =============================================================================

# For now, we only define placeholders. 3D training requires more systems.
# NOTE: With only 4 systems, this split is limited. Consider adding more 3D systems.
TRAIN_SYSTEMS_3D: List[Type[DynamicalSystem]] = [
    Rossler,
    ChenSystem,
]

TEST_SYSTEMS_3D: List[Type[DynamicalSystem]] = [
    Lorenz,  # Canonical benchmark - always held out
    SIRModel,
]


# =============================================================================
# Helper Functions
# =============================================================================


def get_split(dimension: int) -> Tuple[List[Type[DynamicalSystem]], List[Type[DynamicalSystem]]]:
    """
    Get train/test split for a given dimension.

    Parameters
    ----------
    dimension : int
        State space dimension (2 or 3).

    Returns
    -------
    train_systems : List[Type[DynamicalSystem]]
        System classes for training.
    test_systems : List[Type[DynamicalSystem]]
        System classes for testing (held out).

    Raises
    ------
    ValueError
        If dimension is not 2 or 3.
    """
    if dimension == 2:
        return TRAIN_SYSTEMS_2D.copy(), TEST_SYSTEMS_2D.copy()
    elif dimension == 3:
        return TRAIN_SYSTEMS_3D.copy(), TEST_SYSTEMS_3D.copy()
    else:
        raise ValueError(
            f"No train/test split defined for dimension {dimension}. Only 2D and 3D are supported."
        )


def get_all_systems(dimension: int) -> List[Type[DynamicalSystem]]:
    """
    Get all systems for a given dimension.

    Parameters
    ----------
    dimension : int
        State space dimension (2 or 3).

    Returns
    -------
    systems : List[Type[DynamicalSystem]]
        All system classes for the given dimension.
    """
    train, test = get_split(dimension)
    return train + test


def get_split_info() -> Dict[str, Dict[str, List[str]]]:
    """
    Get summary information about the train/test splits.

    Returns
    -------
    info : Dict
        Dictionary with split information for each dimension.
    """
    return {
        "2D": {
            "train": [cls.__name__ for cls in TRAIN_SYSTEMS_2D],
            "test": [cls.__name__ for cls in TEST_SYSTEMS_2D],
            "n_train": len(TRAIN_SYSTEMS_2D),
            "n_test": len(TEST_SYSTEMS_2D),
        },
        "3D": {
            "train": [cls.__name__ for cls in TRAIN_SYSTEMS_3D],
            "test": [cls.__name__ for cls in TEST_SYSTEMS_3D],
            "n_train": len(TRAIN_SYSTEMS_3D),
            "n_test": len(TEST_SYSTEMS_3D),
            "note": "3D training deferred - insufficient systems for meaningful split",
        },
    }


def print_split_info():
    """Print a summary of the train/test splits."""
    info = get_split_info()

    print("=" * 60)
    print("SC-SINDy Train/Test Split Summary")
    print("=" * 60)

    for dim, data in info.items():
        print(f"\n{dim} Systems:")
        print(f"  Train ({data['n_train']}): {', '.join(data['train'])}")
        print(f"  Test ({data['n_test']}): {', '.join(data['test'])}")
        if "note" in data:
            print(f"  Note: {data['note']}")
