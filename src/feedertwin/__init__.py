"""FeederTwin — имитационная модель системы поштучной подачи мелкоразмерных изделий."""

from feedertwin.flowsim import (
    FlowMetrics,
    FlowParams,
    ReplicationStats,
    run_replications,
    run_simulation,
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

__version__ = "0.2.0"

__all__ = [
    "FlowMetrics",
    "FlowParams",
    "ParticleTrace",
    "ReplicationStats",
    "TrayParams",
    "gamma",
    "is_steady_regime",
    "mean_velocity",
    "regime",
    "run_replications",
    "run_simulation",
    "simulate_particle",
    "sweep",
    "__version__",
]
