"""FusionNode: routes messages to layers and collects their outputs.

Build it up one layer at a time as each test step passes:

    node = FusionNode(cfg, layers=[AttitudeLayer(cfg)])          # after step 3
    node = FusionNode(cfg, layers=[AttitudeLayer(cfg), PoseEKF(cfg)])  # step 6

A layer receives a message type by defining a method named after it:
`on_ImuSample(self, msg)`, `on_VoPose(self, msg)`, and so on. Layers can read
each other's outputs through `self.node.state`, which is how the EKF gets
roll/pitch from the attitude layer without the two being coupled.
"""
from __future__ import annotations


class Layer:
    """Base class for a fusion layer."""

    name = "layer"
    #: config keys (dotted) that must be measured before this layer can run
    requires: tuple[str, ...] = ()

    def __init__(self, cfg: dict):
        from .config import require
        self.cfg = cfg
        require(cfg, *self.requires)
        self.node: FusionNode | None = None

    def state(self) -> dict:
        """Outputs of this layer, merged into node.state after each message."""
        return {}


class FusionNode:
    def __init__(self, cfg: dict, layers: list[Layer]):
        self.cfg = cfg
        self.layers = layers
        self.state: dict = {}
        self.t = None
        self._routes: dict[str, list] = {}
        for layer in layers:
            layer.node = self

    def _handlers(self, type_name):
        if type_name not in self._routes:
            self._routes[type_name] = [getattr(l, f"on_{type_name}") for l in self.layers
                                       if hasattr(l, f"on_{type_name}")]
        return self._routes[type_name]

    def process(self, msg) -> dict:
        """Feed one message. Returns the merged state (may be unchanged)."""
        self.t = msg.t
        handlers = self._handlers(type(msg).__name__)
        for h in handlers:
            h(msg)
        if handlers:
            for layer in self.layers:
                self.state.update(layer.state())
            self.state["t"] = self.t
        return self.state

    def run(self, source, on_state=None):
        """Process a whole Source (live or replay)."""
        for msg in source:
            st = self.process(msg)
            if on_state:
                on_state(msg, st)
        return self.state
