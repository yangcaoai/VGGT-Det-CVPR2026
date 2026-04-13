import torch
from einops import reduce
from jaxtyping import Float, Int64
from torch import Tensor

from typing import List, Dict, Set, Tuple
from typing import Union

def sample_discrete_distribution(
    pdf: Float[Tensor, "*batch bucket"],
    num_samples: int,
    eps: float = torch.finfo(torch.float32).eps,
) -> Tuple[
    Int64[Tensor, "*batch sample"],  # index
    Float[Tensor, "*batch sample"],  # probability density
]:
    *batch, bucket = pdf.shape
    normalized_pdf = pdf / (eps + reduce(pdf, "... bucket -> ... ()", "sum"))
    cdf = normalized_pdf.cumsum(dim=-1)
    samples = torch.rand((*batch, num_samples), device=pdf.device) # (bs, num_view, h*w, num_surface, num_samples)
    index = torch.searchsorted(cdf, samples, right=True).clip(max=bucket - 1)
    return index, normalized_pdf.gather(dim=-1, index=index)


def gather_discrete_topk(
    pdf: Float[Tensor, "*batch bucket"],
    num_samples: int,
    eps: float = torch.finfo(torch.float32).eps,
) -> Tuple[
    Int64[Tensor, "*batch sample"],  # index
    Float[Tensor, "*batch sample"],  # probability density
]:
    normalized_pdf = pdf / (eps + reduce(pdf, "... bucket -> ... ()", "sum"))
    index = pdf.topk(k=num_samples, dim=-1).indices
    return index, normalized_pdf.gather(dim=-1, index=index)
