"""
Train/Test splits for factorized network evaluation.

This module provides system splits designed for training and evaluating
the dimension-agnostic factorized structure network. The splits ensure:
1. Test systems have distinct structures from training
2. Canonical benchmarks (like Lorenz) are always held out
3. Proper coverage of structural patterns across dimensions
"""

from typing import Dict, List, Type

from sc_sindy.systems.base import DynamicalSystem

# 2D Systems
from sc_sindy.systems.oscillators import (
    VanDerPol,
    DuffingOscillator,
    DampedHarmonicOscillator,
    ForcedOscillator,
)
from sc_sindy.systems.biological import (
    SelkovGlycolysis,
    CoupledBrusselator,
    SIRModel,
)
from sc_sindy.systems.ecological import (
    CompetitiveExclusion,
    MutualismModel,
    SISEpidemic,
    PredatorPreyTypeII,
    SimplePredatorPrey,
)
from sc_sindy.systems.neural import (
    FitzHughNagumo,
    MorrisLecar,
    HindmarshRose2D,
)
from sc_sindy.systems.canonical import (
    HopfNormalForm,
    CubicOscillator,
    QuadraticOscillator,
    RayleighOscillator,
    LinearOscillator,
)

# 3D Systems
from sc_sindy.systems.chaotic import Lorenz, Rossler, ChenSystem
from sc_sindy.systems.chaotic_3d import (
    ThomasAttractor,
    HalvorsenAttractor,
    SprottB,
    SprottD,
    RabinovichFabrikant,
    AizawaAttractor,
)

# 4D Systems
from sc_sindy.systems.coupled_4d import (
    CoupledVanDerPol,
    CoupledDuffing,
    HyperchaoticLorenz,
    HyperchaoticRossler,
    LotkaVolterra4D,
    CoupledFitzHughNagumo,
    MixedCoupledOscillator,
    LorenzExtended4D,
    SimpleQuadratic4D,
    Cubic4DSystem,
)


# =============================================================================
# 2D SPLITS
# =============================================================================

TRAIN_SYSTEMS_2D_FACTORIZED: List[Type[DynamicalSystem]] = [
    # Oscillators (4 systems)
    VanDerPol,
    DuffingOscillator,
    RayleighOscillator,
    CubicOscillator,
    # Biological/Chemical (2 systems)
    SelkovGlycolysis,
    CoupledBrusselator,
    # Ecological with xy interaction (3 systems)
    CompetitiveExclusion,
    MutualismModel,
    SISEpidemic,
    # Neural (2 systems)
    FitzHughNagumo,
    MorrisLecar,
    # Canonical (1 system)
    HopfNormalForm,
]

TEST_SYSTEMS_2D_FACTORIZED: List[Type[DynamicalSystem]] = [
    # Linear systems (tests generalization to simpler dynamics)
    DampedHarmonicOscillator,
    LinearOscillator,
    ForcedOscillator,
    # Tests xy interaction generalization
    PredatorPreyTypeII,
    # Tests neural dynamics generalization
    HindmarshRose2D,
]

HELDOUT_SYSTEMS_2D_FACTORIZED: List[Type[DynamicalSystem]] = [
    # Additional validation (not used in train or test)
    SimplePredatorPrey,
    QuadraticOscillator,
]


# =============================================================================
# 3D SPLITS
# =============================================================================

TRAIN_SYSTEMS_3D_FACTORIZED: List[Type[DynamicalSystem]] = [
    # Existing 3D chaotic
    Rossler,
    ChenSystem,
    # New 3D chaotic (diverse structures)
    HalvorsenAttractor,  # Quadratic self-interaction (xx, yy, zz)
    SprottB,             # Minimal bilinear (yz, xy)
    AizawaAttractor,     # Mixed xz, yz interactions
]

TEST_SYSTEMS_3D_FACTORIZED: List[Type[DynamicalSystem]] = [
    # Canonical benchmark - NEVER trained on
    Lorenz,
    # Epidemiological - different domain
    SIRModel,
    # Rich bilinear structure
    RabinovichFabrikant,
]

HELDOUT_SYSTEMS_3D_FACTORIZED: List[Type[DynamicalSystem]] = [
    # Additional validation
    ThomasAttractor,
    SprottD,
]


# =============================================================================
# 4D SPLITS
# =============================================================================

TRAIN_SYSTEMS_4D_FACTORIZED: List[Type[DynamicalSystem]] = [
    # Coupled oscillators (cubic terms)
    CoupledVanDerPol,
    CoupledDuffing,
    # Hyperchaotic (bilinear terms)
    HyperchaoticLorenz,
    # Multi-species competition (many bilinear terms)
    LotkaVolterra4D,
    # Mixed coupling (Van der Pol + Duffing)
    MixedCoupledOscillator,
    # Lorenz extension (tests xy coupling)
    LorenzExtended4D,
    # Simple sparse structure
    SimpleQuadratic4D,
    # Pure cubic structure
    Cubic4DSystem,
]

TEST_SYSTEMS_4D_FACTORIZED: List[Type[DynamicalSystem]] = [
    # Different hyperchaotic structure
    HyperchaoticRossler,
    # Coupled neural dynamics
    CoupledFitzHughNagumo,
]

HELDOUT_SYSTEMS_4D_FACTORIZED: List[Type[DynamicalSystem]] = []


# =============================================================================
# COMBINED SPLITS
# =============================================================================

def get_factorized_train_systems() -> Dict[int, List[Type[DynamicalSystem]]]:
    """
    Get all training systems organized by dimension.

    Returns
    -------
    systems_by_dim : Dict[int, List[Type]]
        Dictionary mapping dimension to list of system classes.
    """
    return {
        2: TRAIN_SYSTEMS_2D_FACTORIZED,
        3: TRAIN_SYSTEMS_3D_FACTORIZED,
        4: TRAIN_SYSTEMS_4D_FACTORIZED,
    }


def get_factorized_test_systems() -> Dict[int, List[Type[DynamicalSystem]]]:
    """
    Get all test systems organized by dimension.

    Returns
    -------
    systems_by_dim : Dict[int, List[Type]]
        Dictionary mapping dimension to list of system classes.
    """
    return {
        2: TEST_SYSTEMS_2D_FACTORIZED,
        3: TEST_SYSTEMS_3D_FACTORIZED,
        4: TEST_SYSTEMS_4D_FACTORIZED,
    }


def get_factorized_heldout_systems() -> Dict[int, List[Type[DynamicalSystem]]]:
    """
    Get all held-out systems organized by dimension.

    These are completely excluded from training and testing,
    used only for final validation.

    Returns
    -------
    systems_by_dim : Dict[int, List[Type]]
        Dictionary mapping dimension to list of system classes.
    """
    return {
        2: HELDOUT_SYSTEMS_2D_FACTORIZED,
        3: HELDOUT_SYSTEMS_3D_FACTORIZED,
        4: HELDOUT_SYSTEMS_4D_FACTORIZED,
    }


def print_split_summary():
    """Print a summary of the train/test splits."""
    print("=" * 60)
    print("FACTORIZED NETWORK TRAIN/TEST SPLITS")
    print("=" * 60)

    for dim in [2, 3, 4]:
        train = get_factorized_train_systems().get(dim, [])
        test = get_factorized_test_systems().get(dim, [])
        heldout = get_factorized_heldout_systems().get(dim, [])

        print(f"\n{dim}D SYSTEMS:")
        print(f"  Train ({len(train)}): {[s.__name__ for s in train]}")
        print(f"  Test  ({len(test)}): {[s.__name__ for s in test]}")
        print(f"  Held  ({len(heldout)}): {[s.__name__ for s in heldout]}")

    print("\n" + "=" * 60)
    total_train = sum(len(v) for v in get_factorized_train_systems().values())
    total_test = sum(len(v) for v in get_factorized_test_systems().values())
    total_heldout = sum(len(v) for v in get_factorized_heldout_systems().values())
    print(f"TOTALS: Train={total_train}, Test={total_test}, Held-out={total_heldout}")


if __name__ == "__main__":
    print_split_summary()
