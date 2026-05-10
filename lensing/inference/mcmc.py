"""Posterior sampling helpers (Pyro NUTS).

The thesis ran NUTS with a slightly different prior choice in every notebook;
``run_nuts`` here gives a single helper that takes a Pyro model definition and
returns a flat ``pandas.DataFrame`` of posterior samples - the format used by
``corner.corner`` and reproducible across experiments.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import pandas as pd

try:
    import pyro
    from pyro.infer import MCMC, NUTS
except ImportError:  # pragma: no cover
    pyro = None
    MCMC = None
    NUTS = None


def run_nuts(
    pyro_model: Callable[..., Any],
    *args,
    num_samples: int = 1000,
    warmup_steps: int = 500,
    num_chains: int = 1,
    seed: Optional[int] = 0,
    save_path: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """Run NUTS on a Pyro model and return a DataFrame of posterior samples.

    Parameters
    ----------
    pyro_model : a Pyro model function (signature ``pyro_model(*args, **kwargs)``)
    save_path : if not None, write a CSV to this path - matches the
        ``posterior_samples_*.csv`` pattern used in the thesis
    """
    if pyro is None:
        raise ImportError("Pyro is not installed. `pip install pyro-ppl`.")

    if seed is not None:
        pyro.set_rng_seed(seed)

    kernel = NUTS(pyro_model)
    mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=num_chains)
    mcmc.run(*args, **kwargs)
    samples: Dict[str, Any] = mcmc.get_samples()

    flat = {k: v.detach().cpu().numpy() for k, v in samples.items()}
    df = pd.DataFrame(flat)

    if save_path is not None:
        df.to_csv(save_path, index=False)

    return df
