"""FeederTwin — имитационная модель системы поштучной подачи мелкоразмерных изделий."""

from feedertwin.flowsim import (
    FlowMetrics,
    FlowParams,
    ReplicationStats,
    run_replications,
    run_simulation,
)
from feedertwin.mcda import (
    AhpResult,
    TopsisResult,
    ahp_weights,
    topsis,
    topsis_weight_sweep,
)
from feedertwin.strategies import (
    AdaptiveVibration,
    BaselineThreshold,
    MeteringGate,
)
from feedertwin.transport import (
    ParticleTrace,
    TrayParams,
    gamma,
    is_steady_regime,
    mean_velocity,
    regime,
    simulate_particle,
    sweep,
)

__version__ = "1.0.0"

__all__ = [
    "AdaptiveVibration",
    "AhpResult",
    "BaselineThreshold",
    "FlowMetrics",
    "FlowParams",
    "ParticleTrace",
    "MeteringGate",
    "ReplicationStats",
    "TopsisResult",
    "TrayParams",
    "ahp_weights",
    "gamma",
    "is_steady_regime",
    "mean_velocity",
    "regime",
    "run_replications",
    "run_simulation",
    "simulate_particle",
    "sweep",
    "topsis",
    "topsis_weight_sweep",
    "__version__",
]
