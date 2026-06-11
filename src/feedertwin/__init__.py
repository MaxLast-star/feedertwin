"""FeederTwin — имитационная модель системы поштучной подачи мелкоразмерных изделий."""

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

__version__ = "0.1.0"

__all__ = [
    "ParticleTrace",
    "TrayParams",
    "gamma",
    "is_steady_regime",
    "mean_velocity",
    "regime",
    "simulate_particle",
    "sweep",
    "__version__",
]
