"""Parameter transforms shared across lens and light models.

* :func:`e1e2_to_q_pa` / :func:`q_pa_to_e1e2` — convert between the
  ``(e1, e2)`` ellipticity components used during optimization (smooth,
  no gimbal lock at ``e=0``) and the ``(q, PA)`` axis-ratio + position-
  angle pair used for reporting.
* :func:`sersic_bn` — Ciotti & Bertin (1999) polynomial expansion of
  the Sérsic ``b_n`` constant, accurate to ~1e-4 for ``n ≥ 0.36`` — used
  inside autograd loops where iterating an incomplete-gamma equation
  would be costly.
* :func:`softplus_inverse`, :func:`wrap_angle` — small helpers used by
  the constraint-projection logic in lens / light models.
"""
from .parameters import (
    e1e2_to_q_pa,
    q_pa_to_e1e2,
    sersic_bn,
    softplus_inverse,
    wrap_angle,
)

__all__ = [
    "e1e2_to_q_pa",
    "q_pa_to_e1e2",
    "sersic_bn",
    "softplus_inverse",
    "wrap_angle",
]
