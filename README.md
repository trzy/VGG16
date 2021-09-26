# VGG-16 Inference with PyTorch

*Copyright 2020-2021 Bart Trzynadlowski*

## Description

This is a PyTorch implementation of the VGG-16 image classification model. It
can perform forward inference only and is compatible with weights from the Keras
version of this project.

Please see the `master` branch for more information.

## Usage

Make sure you are running at least Python 3.8 and install the required PIP
packages listed in `requirements.txt`.

Once in your Python environment of choice:

  ```
  pip install -r requirements.txt
  ```

Run the script from the repository directory:

  ```
  python -m VGG16 --load-from=vgg16_weights.hdf5 --show-objects=image.png
  ```

Both weights and an image file must be specified. The image can also be specified as a URL.

##
