from dataclasses import dataclass

from .view_sampler import ViewSamplerCfg
from typing import List, Dict, Set, Tuple
from typing import Union

@dataclass
class DatasetCfgCommon:
    image_shape: List[int]
    background_color: List[float]
    cameras_are_circular: bool
    overfit_to_scene: Union[str, None]
    view_sampler: ViewSamplerCfg
