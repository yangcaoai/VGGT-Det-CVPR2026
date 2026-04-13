from typing import Optional

from .encoder import Encoder
from .encoder_epipolar import EncoderEpipolar, EncoderEpipolarCfg
from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_epipolar import EncoderVisualizerEpipolar

from typing import List, Dict, Set, Tuple
from typing import Union

ENCODERS = {
    "epipolar": (EncoderEpipolar, EncoderVisualizerEpipolar),
}

EncoderCfg = EncoderEpipolarCfg


def get_encoder(cfg: EncoderCfg) -> Tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
