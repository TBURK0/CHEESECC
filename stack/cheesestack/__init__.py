"""CHEESECC autonomy stack skeleton.

Layers (each a subpackage):
    core          shared types, geometry, time, message bus, node base
    perception    sensors -> understanding of the world
    localization  where the vehicle is and how it moves
    mapping       roadway structure
    prediction    future motion of agents
    planning      perception -> driving decisions
    vehicle       internal understanding of the car
    safety        watchdogs, validation, fault handling, fallback
    control       planned motion -> actuator requests
    comms         CAN / DBC / logging / bus health
    sim           digital twin, replay, scenarios, closed loop
    infra         config, params, telemetry, storage
    ml            datasets, training, validation, drift
"""

__version__ = "0.1.0"
