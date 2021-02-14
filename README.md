# Training VGG-16 on ImageNet with TensorFlow 2 and Keras

*Copyright 2020-2021 Bart Trzynadlowski*

## Description

This is an implementation of the VGG-16 image classification model using
TensorFlow 2 and Keras written in Python. The ImageNet dataset is required for
training and evaluation. Inference can be performed on any image file.

The code is capable of replicating the results of the original paper by
Simonyan and Zisserman. Sample training results can be found in the spreadsheet
included in the repository.

## References

VGG-16 was initially presented as configuration D in this paper:

> Very Deep Convolutional Networks for Large-Scale Image Recognition
> Karen Simonyan, Andrew Zisserman
> ICLR 2015

It is available here: https://arxiv.org/abs/1409.1556

The ImageNet dataset must be procured with a valid academic email address from
the Stanford Vision Lab: http://www.image-net.org/

Rumor has it that Torrents are available but you didn't hear that from me.

## Usage

Make sure you are running at least Python 3.8 and install the required PIP
packages listed in `requirements.txt`. I highly recommend using Anaconda on
Windows or a TensorFlow Docker container on Linux.

Once in your Python environment of choice:

  ```
  pip install -r requirements.txt
  ```

At the time of this writing (February 2021), if you are using an Nvidia RTX
30-series GPU, you may need to edit that file and use `tf-nightly-gpu` for
TensorFlow. Otherwise, you might see obvious numerical issues during training.

Run the script from the repository directory:

  ```
  python -m VGG16
  ```

For usage examples, read the comments at the top of `__main__.py`.

**Good luck!**
