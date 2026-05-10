"""Metadata extractor for strong-lens publications.

Two backends:

* **anthropic**: calls Claude with a structured tool-use schema; uses
  ``cache_control`` on the system prompt so a batch run over 100
  papers pays the (~3k token) instructions once;
* **mock**: deterministic regex / number-extraction fallback for
  offline demo.

The structured output is a :class:`LensRecord` dataclass.

Choice of model
---------------
For metadata extraction the value of a frontier model over a "good
enough" model is small — the schema is well-defined and the LLM mainly
parses numbers from a paragraph. We default to Claude Haiku 4.5
(``claude-haiku-4-5``) for cost; users who want maximum reliability can
pass ``model="claude-sonnet-4-6"``.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence


@dataclass
class LensRecord:
    """One row of the lens metadata catalog."""

    name: Optional[str] = None
    theta_E_arcsec: Optional[float] = None
    sigma_v_kms: Optional[float] = None
    q: Optional[float] = None
    z_L: Optional[float] = None
    z_S: Optional[float] = None
    reference: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Mock backend (regex)
# ---------------------------------------------------------------------------
_PAT_NAME = re.compile(r"\b(SDSSJ\d{4}[+-]\d{4}|J\d{4}[+-]\d{4}|HE \d{4}[+-]?\d{4}|RXJ\s?\d{4}\.?\d*[+-]\d{4})\b")
_PAT_THETA = re.compile(
    r"(?:Einstein\s+radius|theta[_\s]*E|\\theta_E)\s*(?:of|=|\\sim|~|≈|is\s+|equals\s+)?\s*"
    r"([0-9]+\.?[0-9]*)\s*(?:''|″|arcsec|arcsec\.|arc\s*sec)",
    re.IGNORECASE,
)
_PAT_SIGMA = re.compile(
    r"(?:velocity\s+dispersion|\\sigma_v|sigma_v|sigma_\\star)\s*(?:of|=|\\sim|~|≈|is\s+|equals\s+)?\s*"
    r"([0-9]+\.?[0-9]*)\s*(?:km\s*/\s*s|km\s*s\^?-?1)",
    re.IGNORECASE,
)
_PAT_ZL = re.compile(r"(?:z[_\s]*L|z[_\s]*lens|z[_\s]*l|lens\s+(?:at\s+)?redshift)\s*(?:=|\\sim|~|≈|of)?\s*([0-9]+\.[0-9]+)", re.IGNORECASE)
_PAT_ZS = re.compile(r"(?:z[_\s]*S|z[_\s]*source|z[_\s]*s|source\s+(?:at\s+)?redshift)\s*(?:=|\\sim|~|≈|of)?\s*([0-9]+\.[0-9]+)", re.IGNORECASE)
_PAT_Q = re.compile(r"(?:axis\s+ratio|q\s*=|\\axis\\)\s*(?:=|of)?\s*([0-9]+\.[0-9]+)", re.IGNORECASE)


def _mock_extract(abstract: str) -> LensRecord:
    rec = LensRecord(notes="extracted by mock backend (regex)")

    m = _PAT_NAME.search(abstract)
    if m:
        rec.name = m.group(1).strip()
    m = _PAT_THETA.search(abstract)
    if m:
        try:
            rec.theta_E_arcsec = float(m.group(1))
        except ValueError:
            pass
    m = _PAT_SIGMA.search(abstract)
    if m:
        try:
            rec.sigma_v_kms = float(m.group(1))
        except ValueError:
            pass
    m = _PAT_ZL.search(abstract)
    if m:
        try:
            rec.z_L = float(m.group(1))
        except ValueError:
            pass
    m = _PAT_ZS.search(abstract)
    if m:
        try:
            rec.z_S = float(m.group(1))
        except ValueError:
            pass
    m = _PAT_Q.search(abstract)
    if m:
        try:
            rec.q = float(m.group(1))
        except ValueError:
            pass
    return rec


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------
_SYSTEM = """You are a research assistant extracting structured metadata
from gravitational-lensing paper abstracts.

For each abstract you receive, return a JSON object with the keys

  name              : str | null   (e.g. "SDSSJ0029-0055" or null)
  theta_E_arcsec    : float | null (Einstein radius in arcsec)
  sigma_v_kms       : float | null (stellar velocity dispersion in km/s)
  q                 : float | null (axis ratio of the lens, 0..1)
  z_L               : float | null (lens redshift)
  z_S               : float | null (source redshift)
  reference         : str | null   (the paper short reference if present)
  notes             : str | null   (one short sentence of context)

Rules:
* Return ONLY the JSON object, no surrounding text.
* If a value is not stated in the abstract, use null. Do not invent.
* For ranges or uncertainties, return the central value.
* For names use the conventional SDSS / HST / etc. designations.
"""


def _anthropic_extract(
    abstract: str,
    model: str = "claude-haiku-4-5",
    api_key: Optional[str] = None,
) -> LensRecord:
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError("`anthropic` SDK not installed. `pip install anthropic`.") from exc

    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model=model,
        max_tokens=512,
        system=[
            # cache_control on the system prompt amortises the prompt
            # across many calls in a batch run.
            {"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}},
        ],
        messages=[{"role": "user", "content": abstract}],
    )
    raw = "".join(b.text for b in msg.content if getattr(b, "text", None))
    # Extract the first JSON object that appears in the response (the
    # model usually returns JSON only, but we are defensive).
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < 0:
        return LensRecord(notes=f"Claude returned non-JSON: {raw[:80]}...")
    try:
        payload = json.loads(raw[start: end + 1])
    except json.JSONDecodeError:
        return LensRecord(notes=f"Claude returned malformed JSON: {raw[:80]}...")
    return LensRecord(**{k: payload.get(k) for k in LensRecord.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class MetadataExtractor:
    """Driver that selects a backend and processes a list of abstracts.

    Examples
    --------
    >>> ex = MetadataExtractor(backend="mock")
    >>> records = ex.extract_many(abstracts)
    >>> df = pd.DataFrame([r.to_dict() for r in records])
    """

    def __init__(self, backend: str = "mock", model: str = "claude-haiku-4-5",
                 api_key: Optional[str] = None):
        if backend not in {"mock", "anthropic"}:
            raise ValueError(f"backend must be mock|anthropic, got {backend!r}")
        self.backend = backend
        self.model = model
        self.api_key = api_key

    def extract(self, abstract: str) -> LensRecord:
        if self.backend == "mock":
            return _mock_extract(abstract)
        return _anthropic_extract(abstract, model=self.model, api_key=self.api_key)

    def extract_many(self, abstracts: Sequence[str]) -> List[LensRecord]:
        return [self.extract(a) for a in abstracts]


def extract_lens_metadata(
    abstract: str,
    *,
    backend: str = "mock",
    model: str = "claude-haiku-4-5",
    api_key: Optional[str] = None,
) -> LensRecord:
    """One-call helper for a single abstract."""
    return MetadataExtractor(backend=backend, model=model, api_key=api_key).extract(abstract)
