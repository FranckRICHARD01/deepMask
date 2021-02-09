# Accurate and Reliable Brain Extraction in Cortical Malformations

PyTorch Implementation using V-net variant of Fully Convolutional Neural Networks

Authors: [Ravnoor Gill](https://github.com/ravnoor), [Benoit Caldairou](https://github.com/bcaldairou), [Neda Bernasconi](https://noel.bic.mni.mcgill.ca/~noel/people/neda-bernasconi/) and [Andrea Bernasconi](https://noel.bic.mni.mcgill.ca/~noel/people/andrea-bernasconi/)

------------------------

Implementation based on:
Milletari, F., Navab, N., & Ahmadi, S. A. (2016, October). [V-net: Fully convolutional neural networks for volumetric medical image segmentation. In 2016 fourth international conference on 3D vision (3DV) (pp. 565-571). IEEE.


### Please cite:
```
@misc{Gill2021,
  author = {Gill RS, et al},
  title = {Accurate and Reliable Brain Extraction in Cortical Malformations},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/NOEL-MNI/deepMask}},
  doi = {10.5281/zenodo.4521716}
}
```

## Pre-requisites
###TODO: Update version requirements
You need to have following in order for this library to work as expected
1. Python >= 3.5
2. Pytorch >= 1.0.0
3. Nibabel >= 1.14.0


## Installation

```
conda create -n deepMask python=3.8
conda activate deepMask
pip install -r app/requirements.txt
```


## Usage
###TODO: Training and Inference


## License 
(from [https://opensource.org/licenses/BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause)) Copyright 2021 Neuroimaging of Epilepsy Laboratory, McGill University

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
