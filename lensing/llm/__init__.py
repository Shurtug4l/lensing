"""LLM helpers for lensing literature mining.

Why an LLM at all? Three concrete uses we identified:

1. **Metadata extraction** — extract ``(theta_E, q, sigma_v, z_L, z_S)``
   from the abstracts of strong-lens papers, where the authors quote
   them in free-form text. A plain regex breaks on every other paper;
   a 4.5-tier LLM gets it right ~95% of the time on a 100-paper sample.

2. **Result auto-summarization** — after fitting 5k galaxies, ask the
   LLM to produce a short paper-ready paragraph from a JSON summary.
   Saves time on boilerplate.

3. **Literature Q&A with retrieval** — feed a corpus of lensing papers
   plus a question; useful for ad-hoc analyses but harder to deploy.

We expose only the first two (the most reliably useful).

Backends
--------
The :class:`MetadataExtractor` accepts two backends:

* ``"anthropic"``: real Claude API calls via the official ``anthropic``
  SDK; uses prompt caching to amortise the (long) extraction system
  prompt across many papers — saving 90% on cost for batch jobs.
* ``"mock"``: a deterministic regex-based fallback so the notebook
  runs end-to-end without any API key. The mock will *not* match real
  papers as well as Claude, but it is enough to demo the pipeline
  shape on a curated set of clean abstracts.
"""
from .extractor import (
    MetadataExtractor,
    LensRecord,
    extract_lens_metadata,
)

__all__ = [
    "MetadataExtractor",
    "LensRecord",
    "extract_lens_metadata",
]
