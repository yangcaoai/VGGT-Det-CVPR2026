from typing import Any

from ...misc.step_tracker import StepTracker
from ..types import Stage
from .view_sampler import ViewSampler
from .view_sampler_all import ViewSamplerAll, ViewSamplerAllCfg
from .view_sampler_arbitrary import ViewSamplerArbitrary, ViewSamplerArbitraryCfg
from .view_sampler_bounded import ViewSamplerBounded, ViewSamplerBoundedCfg
from .view_sampler_evaluation import ViewSamplerEvaluation, ViewSamplerEvaluationCfg
from typing import List, Dict, Set, Tuple
from typing import Union

VIEW_SAMPLERS: Dict[str, ViewSampler[Any]] = {
    "all": ViewSamplerAll,
    "arbitrary": ViewSamplerArbitrary,
    "bounded": ViewSamplerBounded,
    "evaluation": ViewSamplerEvaluation,
}

ViewSamplerCfg = (Union[
    ViewSamplerArbitraryCfg,
    ViewSamplerBoundedCfg,
    ViewSamplerEvaluationCfg,
    ViewSamplerAllCfg
]
)


def get_view_sampler(
    cfg: ViewSamplerCfg,
    stage: Stage,
    overfit: bool,
    cameras_are_circular: bool,
    step_tracker: Union[StepTracker, None],
) -> ViewSampler[Any]:
    return VIEW_SAMPLERS[cfg.name](
        cfg,
        stage,
        overfit,
        cameras_are_circular,
        step_tracker,
    )
