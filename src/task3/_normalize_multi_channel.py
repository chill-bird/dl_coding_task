import json
import torch
from src.constants import (
    TIF_CHANNELS
)

class NormalizeMultiChannel:
    """Normalizes a tensor channel-wise to the range [0, 1]."""
    def __init__(self, stats_json_path: str, eps: float = 1e-6):
        payload = json.loads(open(stats_json_path, "r").read())
        mean = torch.tensor(payload["mean"], dtype=torch.float32)
        std = torch.tensor(payload["std"], dtype=torch.float32).clamp_min(eps)

        self.mean = mean[:, None, None]
        self.std = std[:, None, None]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
       
        selected_mean = self.mean[TIF_CHANNELS] 
        selected_std = self.std[TIF_CHANNELS]

        selected_mean = selected_mean.view(-1, 1, 1)
        mean = selected_mean.to(device=x.device, dtype=x.dtype)
        selected_std = selected_std.view(-1, 1, 1)
        std = selected_std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std
