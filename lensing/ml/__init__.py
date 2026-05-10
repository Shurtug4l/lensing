"""Machine-learning helpers built on top of the package's lens / light models.

* ``datasets``  - synthetic dataset generators (lens vs no-lens, parameter
                  regression targets) ready to plug into ``torch.utils.data``.
* ``models``    - small CNN / DNN / U-Net architectures suited to galaxy-image
                  inputs.
* ``train``     - generic training loop with metrics, AMP, checkpoints.
"""
from . import datasets, models, train

__all__ = ["datasets", "models", "train"]
