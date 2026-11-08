"""Vehicle parameters used by the control layer.

Mirrors the fields of the app's `CarModel` (theapp/CHEESECC/Data/Catalog/CarModel.swift)
that control needs, converted to SI. Defaults are the first entry in the app's
car catalog (wheelbase 2.83 m, steering ratio 12.3, 500 deg max handwheel).

Sign convention inside the control layer: positive curvature / road-wheel angle /
handwheel angle = turn LEFT (CCW), matching CONVENTIONS.md (y left, yaw CCW).
The app's DriveCommand uses compass bearings, so its positive curvature and
steerWheelDeg mean turn RIGHT. The flip happens in exactly one place:
`curvature_control.make_control_command`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class VehicleParams:
    wheelbase_m: float = 2.83
    steering_ratio: float = 12.3  # handwheel deg per road-wheel deg
    max_steer_wheel_deg: float = 500.0
    mass_kg: float = 1475.0
    horsepower: float = 192.0
    torque_nm: float = 260.0
    tire_radius_m: float = 0.328
    drag_coefficient: float = 0.30
    frontal_area_m2: float = 2.27
    # Dynamic bicycle model (used by LQR / understeer). Rough mid-size sedan values.
    cg_to_front_m: float = 1.20  # a
    cornering_stiffness_front: float = 80000.0  # N/rad (both tires)
    cornering_stiffness_rear: float = 90000.0  # N/rad
    yaw_inertia_kgm2: float = 2500.0
    max_brake_decel: float = 6.5  # m/s^2 at brake=1 (LongitudinalModel.swift)

    @property
    def cg_to_rear_m(self) -> float:
        return self.wheelbase_m - self.cg_to_front_m

    @property
    def max_road_wheel_rad(self) -> float:
        return math.radians(self.max_steer_wheel_deg / self.steering_ratio)

    @property
    def max_curvature(self) -> float:
        """Kinematic curvature at full lock (1/m), rear-axle reference."""
        return math.tan(self.max_road_wheel_rad) / self.wheelbase_m

    @property
    def understeer_gradient(self) -> float:
        """K_us in rad/(m/s^2): delta = L*kappa + K_us * a_y (Gillespie)."""
        a, b, L = self.cg_to_front_m, self.cg_to_rear_m, self.wheelbase_m
        wf = self.mass_kg * b / L  # front axle load, kg
        wr = self.mass_kg * a / L
        return wf / self.cornering_stiffness_front - wr / self.cornering_stiffness_rear

    @classmethod
    def from_car_model(cls, car: dict) -> "VehicleParams":
        """Build from a cars.json entry (CarModel field names, mm/kg units)."""
        L = car["wheelbaseMM"] / 1000.0
        return cls(
            wheelbase_m=L,
            steering_ratio=car["steeringRatio"],
            max_steer_wheel_deg=car["maxSteerWheelDeg"],
            mass_kg=car["curbWeightKG"],
            horsepower=car.get("horsepower", 192.0),
            torque_nm=car.get("torqueNm", 260.0),
            tire_radius_m=car.get("tireRadiusM", 0.328),
            drag_coefficient=car.get("dragCoefficient", 0.30),
            frontal_area_m2=car.get("frontalAreaM2", 2.27),
            cg_to_front_m=0.42 * L,
            # TODO(control): identify cornering stiffness / inertia per car instead of scaling.
            cornering_stiffness_front=80000.0 * car["curbWeightKG"] / 1475.0,
            cornering_stiffness_rear=90000.0 * car["curbWeightKG"] / 1475.0,
            yaw_inertia_kgm2=2500.0 * car["curbWeightKG"] / 1475.0,
        )
