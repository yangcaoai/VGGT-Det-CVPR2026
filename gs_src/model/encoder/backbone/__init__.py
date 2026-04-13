from typing import Any

from .backbone import Backbone
from .backbone_dino import BackboneDino, BackboneDinoCfg
from .backbone_resnet import BackboneResnet, BackboneResnetCfg
from typing import List, Dict, Set, Tuple
from typing import Union

BACKBONES: Dict[str, Backbone[Any]] = {
    "resnet": BackboneResnet,
    "dino": BackboneDino,
}

BackboneCfg = Union[BackboneResnetCfg, BackboneDinoCfg]


def get_backbone(cfg: BackboneCfg, d_in: int) -> Backbone[Any]:
    return BACKBONES[cfg.name](cfg, d_in)
