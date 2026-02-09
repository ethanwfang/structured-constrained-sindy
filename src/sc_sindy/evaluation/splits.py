"""
Train/test split definitions for proper SC-SINDy evaluation.

This module defines which systems are used for training vs testing the structure
network. The key principle is that TEST systems should never be seen during training
to ensure fair evaluation of generalization.

IMPORTANT: LotkaVolterra is EXCLUDED from both training and testing to enable
unbiased real-world validation on Lynx-Hare predator-prey data.
"""

from typing import Dict, List, Tuple, Type

from ..systems import (
    # Base
    DynamicalSystem,
    # Oscillators
    VanDerPol,
    DuffingOscillator,
    DampedHarmonicOscillator,
    ForcedOscillator,
    # Biological (excluding LotkaVolterra)
    SelkovGlycolysis,
    CoupledBrusselator,
    SIRModel,
    # Ecological (with xy interaction - critical for learning interaction patterns)
    CompetitiveExclusion,
    MutualismModel,
    SISEpidemic,
    PredatorPreyTypeII,
    SimplePredatorPrey,
    # Neural
    FitzHughNagumo,
    MorrisLecar,
    HindmarshRose2D,
    # Canonical
    HopfNormalForm,
    CubicOscillator,
    QuadraticOscillator,
    RayleighOscillator,
    LinearOscillator,
    # 3D Chaotic
    Lorenz,
    Rossler,
    ChenSystem,
)

# =============================================================================
# 2D System Splits
# =============================================================================
# Training: 14 systems with diverse structural patterns
# Testing: 5 systems held out for evaluation
# Real-world: Lynx-Hare (LotkaVolterra-like) - completely excluded

TRAIN_SYSTEMS_2D: List[Type[DynamicalSystem]] = [
    # Oscillators (no xy)
    VanDerPol,
    DuffingOscillator,
    RayleighOscillator,
    CubicOscillator,
    # Biological (no xy)
    SelkovGlycolysis,
    CoupledBrusselator,
    # Ecological WITH xy interaction (critical for Lynx-Hare generalization)
    CompetitiveExclusion,
    MutualismModel,
    SISEpidemic,
    SimplePredatorPrey,
    # Neural
    FitzHughNagumo,
    MorrisLecar,  # has xy
    # Canonical
    HopfNormalForm,
    QuadraticOscillator,
]

TEST_SYSTEMS_2D: List[Type[DynamicalSystem]] = [
    # Held out for testing generalization
    DampedHarmonicOscillator,
    ForcedOscillator,
    LinearOscillator,
    HindmarshRose2D,
    PredatorPreyTypeII,  # has xy - tests xy generalization
]

# Systems with xy interaction available for training (NOT including LotkaVolterra)
XY_TRAIN_SYSTEMS_2D: List[Type[DynamicalSystem]] = [
    CompetitiveExclusion,
    MutualismModel,
    SISEpidemic,
    SimplePredatorPrey,
    MorrisLecar,
]

# =============================================================================
# 3D System Splits
# =============================================================================

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
    Get all systems for a given dimension (excluding LotkaVolterra).

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


def get_xy_training_systems() -> List[Type[DynamicalSystem]]:
    """
    Get 2D systems with xy interaction term for training.

    These systems are critical for learning bilinear interaction patterns
    that generalize to Lotka-Volterra-like dynamics (e.g., Lynx-Hare data).

    Returns
    -------
    systems : List[Type[DynamicalSystem]]
        System classes with xy interaction term.
    """
    return XY_TRAIN_SYSTEMS_2D.copy()


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
            "xy_train": [cls.__name__ for cls in XY_TRAIN_SYSTEMS_2D],
            "n_train": len(TRAIN_SYSTEMS_2D),
            "n_test": len(TEST_SYSTEMS_2D),
            "n_xy_train": len(XY_TRAIN_SYSTEMS_2D),
            "note": "LotkaVolterra excluded for unbiased Lynx-Hare validation",
        },
        "3D": {
            "train": [cls.__name__ for cls in TRAIN_SYSTEMS_3D],
            "test": [cls.__name__ for cls in TEST_SYSTEMS_3D],
            "n_train": len(TRAIN_SYSTEMS_3D),
            "n_test": len(TEST_SYSTEMS_3D),
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
        if "xy_train" in data:
            print(f"  XY Training ({data['n_xy_train']}): {', '.join(data['xy_train'])}")
        if "note" in data:
            print(f"  Note: {data['note']}")
