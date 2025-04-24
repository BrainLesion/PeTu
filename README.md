# PeTu

[![Python Versions](https://img.shields.io/pypi/pyversions/petu)](https://pypi.org/project/petu/)
[![Stable Version](https://img.shields.io/pypi/v/petu?label=stable)](https://pypi.python.org/pypi/petu/)
[![Documentation Status](https://readthedocs.org/projects/petu/badge/?version=latest)](http://petu.readthedocs.io/?badge=latest)
[![tests](https://github.com/BrainLesion/petu/actions/workflows/tests.yml/badge.svg)](https://github.com/BrainLesion/petu/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/BrainLesion/petu/graph/badge.svg?token=A7FWUKO9Y4)](https://codecov.io/gh/BrainLesion/petu)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Description 
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

If you use `petu` in your research, please cite it to support the development!

```
TODO: citation will be added asap
```

## Contributing

We welcome all kinds of contributions from the community!

### Reporting Bugs, Feature Requests and Questions

Please open a new issue [here](https://github.com/BrainLesion/petu/issues).

### Code contributions

Nice to have you on board! Please have a look at our [CONTRIBUTING.md](CONTRIBUTING.md) file.
