from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor
from typing import List, Dict, Set, Tuple
from typing import Union

T_cfg = TypeVar("T_cfg")
T_encoder = TypeVar("T_encoder")


class EncoderVisualizer(ABC, Generic[T_cfg, T_encoder]):
    cfg: T_cfg
    encoder: T_encoder

    def __init__(self, cfg: T_cfg, encoder: T_encoder) -> None:
        self.cfg = cfg
        self.encoder = encoder

    @abstractmethod
    def visualize(
        self,
        context: dict,
        global_step: int,
    ) -> Dict[str, Float[Tensor, "3 _ _"]]:
        pass
