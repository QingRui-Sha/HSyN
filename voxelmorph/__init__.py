# ---- voxelmorph ----
# unsupervised learning for image registration

import os

# set version
__version__ = '0.2'


from packaging import version

# ensure valid neurite version is available
import neurite
minv = '0.2'
curv = getattr(neurite, '__version__', None)
if curv is None or version.parse(curv) < version.parse(minv):
    raise ImportError(f'voxelmorph requires neurite version {minv} or greater, '
                      f'but found version {curv}')

# move on the actual voxelmorph imports

from . import datasets
from . import functions
from . import py
from .py.utils import default_unet_features


# import backend-dependent submodules
backend = py.utils.get_backend()
os.environ['NEURITE_BACKEND'] = 'pytorch'

from . import torch
from .torch import layers
from .torch import networks
from .torch import losses
