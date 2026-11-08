"""Layer 3: what is around the car, from the depth camera and ultrasonics.
(Step 9, after the safety layer from step 7 is in place on the MCU)

Two outputs, built in this order:

  1. clear_distance(): the free distance ahead along the path the car is
     about to sweep, from RAW readings. Simple, testable, and what the
     planner and safety checks need first.
       - camera: depth points inside the corridor (car width + margin)
       - ultrasonic: an echo counts if any part of its beam arc falls inside
         the corridor (beam width from step 1)
       - combine by taking the minimum (conservative); weight by measured
         noise only once you have the numbers from step 1

  2. An occupancy grid (log-odds) for planning around things, once 1 works:
       - camera rays: cells along the ray free, the hit cell occupied
       - ultrasonic: cells inside the cone free up to the range, cells on the
         arc at the range weakly occupied (the echo could be anywhere on it)
     Needs the pose EKF, so it comes after step 6.

Test order (step 9): posts beside the path must NOT stop the car; posts on
the path must; then a suddenly appearing obstacle; then glass/black objects
that only the ultrasonics see.
"""
from __future__ import annotations

import numpy as np

from ..node import Layer


class ObstacleLayer(Layer):
    name = "obstacles"
    requires = ("vehicle.width", "vehicle.front_bumper_x",
                "ultrasonics.beam_half_angle_deg", "ultrasonics.max_range")

    def __init__(self, cfg):
        super().__init__(cfg)
        self.us = {s["name"]: s for s in cfg["ultrasonics"]["sensors"]}
        self.us_latest = {}           # name -> (t, range or None)
        self.cam_points = np.zeros((0, 2))

    def on_UltrasonicReading(self, msg):
        self.us_latest[msg.sensor] = (msg.t, msg.range_m)

    def on_depth_points(self, t, pts_body):
        """Call with (M,2) body-frame obstacle points from your depth pipeline.
        TODO (step 9): depth image -> 3D points -> keep the height band above
        the floor and below the car's roof -> body-frame xy."""
        self.cam_points = np.asarray(pts_body, float).reshape(-1, 2)

    def clear_distance(self, curvature=0.0):
        """TODO (step 9, output 1): free distance ahead along the corridor."""
        raise NotImplementedError("Step 9: corridor clearance")

    def state(self):
        return {}
