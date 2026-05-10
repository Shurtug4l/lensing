# Contributing

Thanks for your interest. This is a personal research project but I welcome
issues and pull requests.

## Getting started

```bash
git clone https://github.com/Shurtug4l/lensing.git
cd lensing
pip install -r requirements.txt
pytest tests/
```

All 30 tests should pass in ~10 s.

## Code style

* Python ≥ 3.10, type hints encouraged where they clarify intent.
* Docstrings should explain the **why** as well as the **what** —
  the lens-equation formulas are standard, the NaN-avoiding tricks
  are not.
* No emojis in source files unless explicitly requested.
* New notebooks are generated via `scripts/_make_notebooks.py`; do not
  hand-edit the `.ipynb` JSON.

## Running individual tests

```bash
pytest tests/test_advanced_lenses.py -v       # specific file
pytest -k 'sersic and recover'                # by name
pytest --tb=short                             # shorter tracebacks
```

## Adding a new module

See the **"How to extend"** section in `docs/usage.md`. The general
recipe:

1. Create the file under `lensing/<subpackage>/<name>.py` with a
   ``nn.Module`` exposing the relevant API.
2. Re-export from the sub-package's `__init__.py`.
3. Add a smoke test in `tests/test_<topic>.py`.
4. (Optional) author a notebook via `_make_notebooks.py` so the
   feature has a worked example.

## Reporting bugs

Open an issue with: a minimal reproducer, the platform / PyTorch
version (`gl.config.setup` + the printed environment block from
notebook 13 are sufficient), and the full traceback.
