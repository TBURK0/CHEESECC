"""Vehicle parameters loaded from the app's car catalog (`cars.json`).

The catalog (``theapp/CHEESECC/Data/Catalog/cars.json``, mirrored by the Swift
``CarModel``) stores dimensions in mm, mass in kg and handwheel limits in
degrees. `VehicleParams` converts everything to SI and fills in quantities the
catalog lacks (CG location, yaw inertia, tire cornering stiffness) from simple,
documented heuristics so the dynamics models have a complete parameter set.

Heuristics (all overridable through `ParamsConfig` or `dataclasses.replace`):
- Static front weight fraction by drivetrain (FWD 0.61, RWD 0.52, AWD 0.57).
- Axle cornering stiffness = coefficient [N/rad per N of axle load] * axle load;
  the rear coefficient is larger so the car is mildly understeering.
- Yaw inertia Iz = k * m * lf * lr (k ~ 1 for passenger cars).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Union

GRAVITY = 9.81
AIR_DENSITY = 1.225

#: Default catalog location, resolved relative to this repo checkout:
#: <repo>/stack/cheesestack/vehicle/params.py -> <repo>/theapp/CHEESECC/Data/Catalog/cars.json
DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[3] / "theapp" / "CHEESECC" / "Data" / "Catalog" / "cars.json"


@dataclass
class ParamsConfig:
    """Heuristics used to derive parameters the catalog does not contain."""

    front_weight_fraction: dict = field(default_factory=lambda: {"FWD": 0.61, "RWD": 0.52, "AWD": 0.57, "4WD": 0.57})
    default_front_weight_fraction: float = 0.58
    payload_kg: float = 80.0  # driver + phone rig on top of curb weight
    cornering_coeff_front: float = 11.0  # N/rad per N of axle load
    cornering_coeff_rear: float = 14.0
    yaw_inertia_factor: float = 1.0
    rolling_resistance: float = 0.013  # matches the Swift LongitudinalModel
    tire_friction: float = 0.9


@dataclass(frozen=True)
class VehicleParams:
    """Complete SI parameter set for one vehicle. Reference point is the rear axle (base_link)."""

    id: str = "generic-sedan"
    name: str = "Generic sedan"
    wheelbase: float = 2.75  # m
    track_front: float = 1.56  # m
    track_rear: float = 1.56  # m
    length: float = 4.75
    width: float = 1.82
    height: float = 1.45
    mass: float = 1500.0  # kg, curb + payload
    steering_ratio: float = 13.0  # handwheel / road-wheel
    max_steer_wheel_deg: float = 500.0
    tire_radius: float = 0.32  # m
    drag_coefficient: float = 0.30
    frontal_area: float = 2.2  # m^2
    horsepower: float = 180.0
    torque_nm: float = 250.0
    drivetrain: str = "FWD"
    front_weight_fraction: float = 0.6
    cornering_stiffness_front: float = 0.0  # N/rad, per axle (0 => derived)
    cornering_stiffness_rear: float = 0.0
    yaw_inertia: float = 0.0  # kg m^2 (0 => derived)
    rolling_resistance: float = 0.013
    tire_friction: float = 0.9

    def __post_init__(self) -> None:
        if self.wheelbase <= 0 or self.mass <= 0 or self.steering_ratio <= 0 or self.tire_radius <= 0:
            raise ValueError(f"non-physical vehicle params for {self.id}")
        if not 0.0 < self.front_weight_fraction < 1.0:
            raise ValueError("front_weight_fraction must be in (0, 1)")
        if self.cornering_stiffness_front <= 0 or self.cornering_stiffness_rear <= 0:
            cfg = ParamsConfig()
            object.__setattr__(self, "cornering_stiffness_front",
                               self.cornering_stiffness_front or cfg.cornering_coeff_front * self.axle_load_front)
            object.__setattr__(self, "cornering_stiffness_rear",
                               self.cornering_stiffness_rear or cfg.cornering_coeff_rear * self.axle_load_rear)
        if self.yaw_inertia <= 0:
            object.__setattr__(self, "yaw_inertia", self.mass * self.lf * self.lr)

    # --- geometry -----------------------------------------------------------
    @property
    def lr(self) -> float:
        """CG to rear axle [m]. Front axle carries m*g*lr/L."""
        return self.front_weight_fraction * self.wheelbase

    @property
    def lf(self) -> float:
        """CG to front axle [m]."""
        return self.wheelbase - self.lr

    @property
    def axle_load_front(self) -> float:
        return self.mass * GRAVITY * self.front_weight_fraction

    @property
    def axle_load_rear(self) -> float:
        return self.mass * GRAVITY * (1.0 - self.front_weight_fraction)

    @property
    def max_road_wheel_angle(self) -> float:
        """Road-wheel angle at full lock [rad]."""
        return math.radians(self.max_steer_wheel_deg / self.steering_ratio)

    @property
    def min_turn_radius(self) -> float:
        """Rear-axle radius at full lock (matches CarModel.rearAxleTurnRadiusM)."""
        t = math.tan(self.max_road_wheel_angle)
        return self.wheelbase / t if t > 0.05 else math.inf

    @property
    def max_curvature(self) -> float:
        return 1.0 / self.min_turn_radius

    # --- handling -----------------------------------------------------------
    @property
    def understeer_gradient(self) -> float:
        """K [rad / (m/s^2)]: delta = L/R + K * a_y. Positive => understeer."""
        return self.mass / self.wheelbase * (self.lr / self.cornering_stiffness_front - self.lf / self.cornering_stiffness_rear)

    @property
    def characteristic_speed(self) -> float:
        """Speed of max yaw-rate gain for an understeering car, sqrt(L/K) [m/s]; inf if K <= 0."""
        k = self.understeer_gradient
        return math.sqrt(self.wheelbase / k) if k > 0 else math.inf

    @property
    def critical_speed(self) -> float:
        """Speed at which an oversteering car becomes unstable, sqrt(-L/K) [m/s]; inf if K >= 0."""
        k = self.understeer_gradient
        return math.sqrt(-self.wheelbase / k) if k < 0 else math.inf

    def steady_state_yaw_rate(self, v: float, road_wheel: float) -> float:
        """r_ss = v * delta / (L + K v^2) (linear, small-angle)."""
        return v * road_wheel / (self.wheelbase + self.understeer_gradient * v * v)

    def handwheel_gain_deg_per_curvature(self) -> float:
        """Small-angle handwheel degrees per 1/m (BicycleModel.theoreticalGain)."""
        return self.wheelbase * self.steering_ratio * 180.0 / math.pi

    # --- construction -------------------------------------------------------
    @classmethod
    def from_catalog_entry(cls, entry: dict, config: Optional[ParamsConfig] = None) -> "VehicleParams":
        """Build from one `cars.json` record (field names as in CarModel.swift)."""
        cfg = config or ParamsConfig()
        drivetrain = str(entry.get("drivetrain", "FWD")).upper()
        wf = cfg.front_weight_fraction.get(drivetrain, cfg.default_front_weight_fraction)
        mass = float(entry["curbWeightKG"]) + cfg.payload_kg
        name = f"{entry.get('year', '')} {entry.get('make', '')} {entry.get('model', '')}".strip()
        p = cls(
            id=str(entry["id"]),
            name=name,
            wheelbase=float(entry["wheelbaseMM"]) / 1000.0,
            track_front=float(entry["frontTrackMM"]) / 1000.0,
            track_rear=float(entry["rearTrackMM"]) / 1000.0,
            length=float(entry["lengthMM"]) / 1000.0,
            width=float(entry["widthMM"]) / 1000.0,
            height=float(entry["heightMM"]) / 1000.0,
            mass=mass,
            steering_ratio=float(entry["steeringRatio"]),
            max_steer_wheel_deg=float(entry["maxSteerWheelDeg"]),
            tire_radius=float(entry["tireRadiusM"]),
            drag_coefficient=float(entry["dragCoefficient"]),
            frontal_area=float(entry["frontalAreaM2"]),
            horsepower=float(entry["horsepower"]),
            torque_nm=float(entry["torqueNm"]),
            drivetrain=drivetrain,
            front_weight_fraction=wf,
            cornering_stiffness_front=cfg.cornering_coeff_front * mass * GRAVITY * wf,
            cornering_stiffness_rear=cfg.cornering_coeff_rear * mass * GRAVITY * (1.0 - wf),
            rolling_resistance=cfg.rolling_resistance,
            tire_friction=cfg.tire_friction,
        )
        if cfg.yaw_inertia_factor != 1.0:
            p = replace(p, yaw_inertia=cfg.yaw_inertia_factor * p.mass * p.lf * p.lr)
        # TODO(vehicle): measured CG / Iz / cornering stiffness per car (axle scales, ramp-steer test).
        return p


PathLike = Union[str, Path]


def load_catalog(path: Optional[PathLike] = None, config: Optional[ParamsConfig] = None) -> dict[str, VehicleParams]:
    """Load every car in the catalog, keyed by catalog `id`. Read-only."""
    p = Path(path) if path is not None else DEFAULT_CATALOG_PATH
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{p}: expected a JSON list of cars")
    out: dict[str, VehicleParams] = {}
    for entry in data:
        vp = VehicleParams.from_catalog_entry(entry, config)
        out[vp.id] = vp
    return out


def load_vehicle_params(car_id: str, path: Optional[PathLike] = None, config: Optional[ParamsConfig] = None) -> VehicleParams:
    """Load one car by `id` (e.g. "honda-accord-2018")."""
    cat = load_catalog(path, config)
    if car_id not in cat:
        raise KeyError(f"unknown car id {car_id!r}; known: {sorted(cat)[:5]}...")
    return cat[car_id]
