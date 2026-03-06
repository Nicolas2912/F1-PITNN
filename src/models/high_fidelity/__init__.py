"""High-fidelity tire model modules (P1-P5)."""

from .adapters import inputs_from_legacy, state_from_legacy, state_to_legacy
from .boundary_conditions import (
    BoundaryConditionModel,
    BoundaryHeatFlows,
    BoundaryState,
    HighFidelityBoundaryParameters,
)
from .materials import ViscoelasticMaterialModel
from .simulator import HighFidelityTireSimulator
from .thermal_solver import ThermalFieldSolver2D, ThermalSolverStepResult
from .types import (
    HighFidelityTireDiagnostics,
    HighFidelityTireInputs,
    HighFidelityTireModelParameters,
    HighFidelityTireState,
)
from .uq import HighFidelityUQ, LHSResult, ParameterPrior, QuantileEnvelope, SobolResult, SobolSensitivityIndex
from .vehicle_simulator import (
    HighFidelityVehicleDiagnostics,
    HighFidelityVehicleSimulator,
    HighFidelityVehicleState,
)
from .wheel_coupling import WheelCouplingResult, WheelForceCouplingModel

__all__ = [
    "HighFidelityTireDiagnostics",
    "HighFidelityTireInputs",
    "HighFidelityTireModelParameters",
    "HighFidelityBoundaryParameters",
    "HighFidelityTireSimulator",
    "HighFidelityTireState",
    "HighFidelityVehicleDiagnostics",
    "HighFidelityVehicleSimulator",
    "HighFidelityVehicleState",
    "HighFidelityUQ",
    "LHSResult",
    "ParameterPrior",
    "QuantileEnvelope",
    "SobolResult",
    "SobolSensitivityIndex",
    "BoundaryConditionModel",
    "BoundaryHeatFlows",
    "BoundaryState",
    "inputs_from_legacy",
    "state_from_legacy",
    "state_to_legacy",
    "ThermalFieldSolver2D",
    "ThermalSolverStepResult",
    "ViscoelasticMaterialModel",
    "WheelCouplingResult",
    "WheelForceCouplingModel",
]
