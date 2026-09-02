from __future__ import annotations

from .config import G6Config, MeridianConfig, NanoConfig
from .inference import ForecastOutput, forecast

class Meridian:
    def __init__(self, config: MeridianConfig = NanoConfig):
        if config.size == "g6":
            raise RuntimeError("Meridian-G6 weights are not shipped.")
        self.config = config

    def predict(self, target, horizon: int, **kwargs) -> ForecastOutput:
        return forecast(target, horizon, config=self.config, **kwargs)

def specify_g6() -> MeridianConfig:
    return G6Config
