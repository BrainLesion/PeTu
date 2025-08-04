# PeTu

[![Python Versions](https://img.shields.io/pypi/pyversions/petu)](https://pypi.org/project/petu/)
[![Stable Version](https://img.shields.io/pypi/v/petu?label=stable)](https://pypi.python.org/pypi/petu/)
[![Documentation Status](https://readthedocs.org/projects/petu/badge/?version=latest)](http://petu.readthedocs.io/?badge=latest)
[![tests](https://github.com/BrainLesion/petu/actions/workflows/tests.yml/badge.svg)](https://github.com/BrainLesion/petu/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/BrainLesion/petu/graph/badge.svg?token=A7FWUKO9Y4)](https://codecov.io/gh/BrainLesion/petu)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

PeTu is a fully automated pipeline for segmenting pediatric brain tumors. It uses a 3D nnU-Net model trained on co-registered multi-parametric MRI scans, including T1c, T1n, T2w, and T2f sequences. Subsequently, the model provides segmented tumor regions, including:

1. T2-hyperintense region (T2H) – typically encompassing solid tumor mass and associated edema.
2. Enhancing tumor (ET) – regions with contrast uptake, indicative of active or aggressive tumor areas.
3. Cystic component (CC) – fluid-filled regions often seen in certain pediatric tumor types.

## Features


## Installation

With a Python 3.10+ environment, you can install `petu` directly from [PyPI](https://pypi.org/project/petu/):

```bash
pip install petu
```


## Use Cases and Tutorials

A minimal example to create a segmentation could look like this:

```python
from petu import Inferer

inferer = Inferer()

# Save NIfTI files
inferer.infer(
    t1c="path/to/t1c.nii.gz",
    fla="path/to/fla.nii.gz",
    t1="path/to/t1.nii.gz",
    t2="path/to/t2.nii.gz",
    ET_segmentation_file="example/ET.nii.gz",
    CC_segmentation_file="example/CC.nii.gz",
    T2H_segmentation_file="example/T2H.nii.gz",
)

# Or directly use pre-loaded NumPy data. (Both outputs work as well)
et, cc, t2h = inferer.infer(
    t1c=t1c_np,
    fla=fla_np,
    t1=t1_np,
    t2=t2_np,
)
```
> [!NOTE]  
>If you're interested in the PeTu package, the [Pediatric Segmentation](https://github.com/BrainLesion/BraTS?tab=readme-ov-file#pediatric-segmentation) may also be of interest.
<!-- For more examples and details please refer to our extensive Notebook tutorials here [NBViewer](https://nbviewer.org/github/BrainLesion/tutorials/blob/main/petu/tutorial.ipynb) ([GitHub](https://github.com/BrainLesion/tutorials/blob/main/petu/tutorial.ipynb)). For the best experience open the notebook in Colab. -->


## Citation
Please support our development by citing the following manuscripts:

[Enhancing efficiency in paediatric brain tumour segmentation using a pathologically diverse single-center clinical dataset](https://doi.org/10.48550/arXiv.2507.22152)

```
@misc{piffer2025enhancingefficiencypaediatricbrain,
      title={Enhancing efficiency in paediatric brain tumour segmentation using a pathologically diverse single-center clinical dataset}, 
      author={A. Piffer and J. A. Buchner and A. G. Gennari and P. Grehten and S. Sirin and E. Ross and I. Ezhov and M. Rosier and J. C. Peeken and M. Piraud and B. Menze and A. Guerreiro Stücklin and A. Jakab and F. Kofler},
      year={2025},
      eprint={2507.22152},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.22152}, 
}
```

## Contributing

We welcome all kinds of contributions from the community!

### Reporting Bugs, Feature Requests and Questions

Please open a new issue [here](https://github.com/BrainLesion/petu/issues).

### Code contributions

Nice to have you on board! Please have a look at our [CONTRIBUTING.md](CONTRIBUTING.md) file.
