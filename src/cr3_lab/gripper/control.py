"""One closure setpoint for the mechanically coupled parallel slides."""
import numpy as np


class ParallelGripper:
    def __init__(self, model):
        actuator = model.actuator("gripper_closure")
        self.actuator = actuator.id
        self.travel_m = float(actuator.ctrlrange[1])

    def command(self, data, closure):
        """Set closure in [0, 1], where 0 is open and 1 is closed."""
        if not np.isfinite(closure) or not 0 <= closure <= 1:
            raise ValueError("Closure must be in [0, 1]")
        data.ctrl[self.actuator] = self.travel_m * closure

    def command_closure(self, data, closure, mode):
        """Rate-limit closing to 0.05 per control tick in gradual mode."""
        if mode == "gradual":
            current = data.ctrl[self.actuator] / self.travel_m
            closure = min(closure, current + 0.05)
        elif mode != "fast":
            raise ValueError(f"Unknown closure mode {mode!r}")
        self.command(data, closure)

    def open(self, data):
        self.command(data, 0.0)

    def close(self, data):
        self.command(data, 1.0)
