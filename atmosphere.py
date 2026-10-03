"""Atmospheric density models used by the ascent simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import RocketConfig

# Specific gas constant for dry air [J/(kg·K)]
R_AIR = 287.05287

# US Standard Atmosphere 1976 lower-atmosphere layers
# (base geopotential altitude [m], base temperature [K], lapse rate [K/m],
#  base pressure [Pa])
_ISA_LAYERS = (
    (0.0, 288.15, -0.0065, 101_325.0),
    (11_000.0, 216.65, 0.0, 22_632.1),
    (20_000.0, 216.65, 0.0010, 5474.89),
    (32_000.0, 228.65, 0.0028, 868.019),
    (47_000.0, 270.65, 0.0, 110.906),
    (51_000.0, 270.65, -0.0028, 66.9389),
    (71_000.0, 214.65, -0.0020, 3.95642),
)


@dataclass
class AtmosphereModel:
    """Altitude-dependent density. Gravity is handled separately in physics.py."""

    config: RocketConfig

    def density(self, altitude_m: np.ndarray | float) -> np.ndarray | float:
        h = np.maximum(np.asarray(altitude_m, dtype=float), 0.0)
        if self.config.atmosphere_model == "exponential":
            rho = self._exponential(h)
        else:
            rho = self._standard(h)
        if np.isscalar(altitude_m):
            return float(np.asarray(rho))
        return rho

    def _exponential(self, h: np.ndarray) -> np.ndarray:
        return self.config.sea_level_density * np.exp(-h / self.config.scale_height)

    def _standard(self, h: np.ndarray) -> np.ndarray:
        """Piecewise ISA 1976 hydrostatic atmosphere, exponential above 86 km."""
        rho = np.empty_like(h, dtype=float)
        g0 = self.config.sea_level_gravity
        for i, alt in enumerate(np.atleast_1d(h).ravel()):
            rho.ravel()[i] = self._isa_density_scalar(float(alt), g0)
        if h.ndim == 0:
            return rho.reshape(())
        return rho.reshape(h.shape)

    def _isa_density_scalar(self, h: float, g0: float) -> float:
        if h < 0.0:
            h = 0.0
        if h > 86_000.0:
            rho_86 = self._isa_density_scalar(86_000.0, g0)
            return float(rho_86 * np.exp(-(h - 86_000.0) / self.config.scale_height))

        layer_index = 0
        for k in range(len(_ISA_LAYERS) - 1, -1, -1):
            if h >= _ISA_LAYERS[k][0]:
                layer_index = k
                break

        h_b, t_b, lapse, p_b = _ISA_LAYERS[layer_index]
        dh = h - h_b
        if abs(lapse) < 1e-12:
            temperature = t_b
            pressure = p_b * np.exp(-g0 * dh / (R_AIR * t_b))
        else:
            temperature = t_b + lapse * dh
            exponent = -g0 / (lapse * R_AIR)
            pressure = p_b * (temperature / t_b) ** exponent
        temperature = max(temperature, 1.0)
        return float(pressure / (R_AIR * temperature))


def calculate_atmospheric_density(altitude_m: float, config: RocketConfig) -> float:
    return float(AtmosphereModel(config).density(altitude_m))
