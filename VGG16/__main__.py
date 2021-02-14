#
# VGG16
# __main__.py
# Copyright 2020-2021 Bart Trzynadlowski
#
# Program for training the VGG-16 model on the ImageNet dataset. Also includes
# evaluation and single-image inference capabilities. Please see dataset.py
# for details on the required directory structure of the ImageNet dataset.
#
# This program replicates the results of the VGG-16 paper (model configuration
# D):
#
#   Very Deep Convolutional Networks for Large-Scale Image Recognition
#   Karen Simonyan, Andrew Zisserman
#   ICLR 2015
#
# Results are logged automatically to out.csv (the destination can be changed)
# and checkpoint files are saved upon loss improvement. Note that validation
# accuracy is the metric actually monitored in the paper.
#
# Usage Examples
# --------------
#
# Training from scratch on a single scale for 90 epochs:
#
#   python -m VGG16 --dataset-dir=c:\data\imagenet\ILSVRC --train --scale=256
#     --epochs=90
#
# Resuming training from saved weights:
#
#   python -m VGG16 --dataset-dir=c:\data\imagenet\ILSVRC --train --epochs=30
#     --load-from=model-checkpoint.hdf5
#
# Training with scale jittering:
#
#   python -m VGG16 --dataset-dir=c:\data\imagenet\ILSVRC --train
#     --scale=256,512 --epochs=90
#
# Training on only two classes:
#
#   python -m VGG16 --dataset-dir=c:\data\imagenet\ILSVRC --train --epochs=90
#     --class-filter=axolotl,gorilla
#
# Training with weights initialized to the pre-trained Keras model and the
# first two convolutional blocks frozen:
#
#   python -m VGG16 --dataset-dir=c:\data\imagenet\ILSVRC --train --epochs=90
#     --load-pretrained-weights
#     --freeze=block1_conv1,block1_conv2,block2_conv1,block2_conv2,block2_conv3
#
# Evaluating the Keras pre-trained model at scale=384:
#
#   python -m VGG16 --dataset-dir=c:\data\imagenet\ILSVRC --eval
#     --load-pretrained-weights --scale=384
#
# Evaluating a custom model at scale=256:
#
#   python -m VGG16 --dataset-dir=c:\data\imagenet\ILSVRC --eval
#     --load-from=my-model.hdf5 --scale=256
#
# Performing inference (i.e., classifying) a single image file at scale=384:
#
#   python -m VGG16 --scale=384 --infer=image.jpg
#
# Classifying an image found online using the pre-trained model:
#
#   python -m VGG16 --infer=http://trzy.org/img/daytona2_scorpio.jpg
#     --load-pretrained-weights
#

from . import utils
from .models import TrainingModel, InferenceModel
from .dataset import ImageNet

import argparse
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras
import time

def load_pretrained_model_weights(model):
  """
  Load pre-trained weights provided by Keras for VGG-16 trained on ImageNet.
  """
  keras_model = tf.keras.applications.VGG16(weights = "imagenet")
  for keras_layer in keras_model.layers:
    weights = keras_layer.get_weights()
    if len(weights) > 0:
      our_layer = [ layer for layer in model.layers if layer.name == keras_layer.name ][0]
      our_layer.set_weights(weights)

def load_image(url, scale, symmetric):
  """
  Load image from URL (or file). If 'symmetric' is true, both dimensions are
  resized to 'scale'. Otherwise, the smallest dimension is set to 'scale' and
  the other is chosen to preserve the original aspect ratio.
  """
  import imageio
  from PIL import Image

  data = imageio.imread(url, pilmode = "RGB")
  original_shape = data.shape
  (height, width, channels) = data.shape
  image = Image.fromarray(data, mode = "RGB")

  if symmetric:
    (height, width, channels) = (scale, scale, 3)
  else:
    # Smallest size resized to scale
    if height < width:
      (height, width, channels) = (scale, int((width / height) * scale), 3)
    else:
      (height, width, channels) = (int((height / width) * scale), scale, 3)

  image = image.resize((width, height), resample = Image.BILINEAR)
  image_data = np.array(image, dtype = np.float32)
  image_data = tf.keras.applications.vgg16.preprocess_input(x = image_data)
  return image_data.reshape((1, height, width, channels)), original_shape

def list_classes():
  """
  Print all ImageNet class names and indices to stdout.
  """
  idx_to_name = ImageNet.get_index_to_class_name()
  print("Index Name")
  print("----- ----")
  for idx, name in idx_to_name.items():
    print("%-5d %s" % (idx, name))

# Returns a class filter containing of a subset of classes to use. If empty,
# all classes will be used.
def construct_class_filter(options):
  """
  Returns a set consisting of class names corresponding to those used by Keras
  or an empty set if no filter is to be applied and all 1000 classes are to be
  used.
  """
  if options.class_filter == None:
    return {}
  filter = set(options.class_filter.split(","))
  return set(ImageNet.get_index_to_class_name(class_filter = filter).values())  # not redundant: ImageNet module will validate filter names for us

def train(model, class_filter, scale_range, validation_scale, options):
  """
  Trains the VGG-16 model.
  """
  # Print parameters and model information
  training_model.summary()
  print("======================")
  print("  Initialized  : %s" % (options.load_from if options.load_from else "default"))
  print("  Save To      : %s" % (options.save_to if options.save_to else "n/a"))
  print("  Class Filter : %s" % (class_filter if len(class_filter) > 0 else "n/a"))
  print("  Classes      : %d" % model.num_classes)
  print("  Learning Rate: %f" % options.learning_rate)
  print("  Momentum     : %f" % options.momentum)
  print("  Dropout      : %f" % options.dropout)
  print("  L2           : %f" % options.l2)
  print("  Batch Size   : %d" % options.batch_size)
  print("  Augmentation : %s" % ("enabled" if options.augment else "disabled"))
  print("  Scale Range  : [%d,%d]" % (min(scale_range), max(scale_range)))
  print("  Val. Scale   : %d" % validation_scale)
  print("======================")

  # Enable Tensorflow profiling
  tf.profiler.experimental.server.start(6009)

  # Add callbacks for model checkpointing and CSV logging
  from tensorflow.keras.callbacks import ReduceLROnPlateau
  from tensorflow.keras.callbacks import ModelCheckpoint
  filepath = "weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
  callbacks_list = []
  callbacks_list.append(tf.keras.callbacks.ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min'))
  callbacks_list.append(utils.CSVLogCallback(filename = options.log, log_epoch_number = True, log_learning_rate = True))

  # Optional callbacks
  if options.patience:
    # Reduce learning rate when validation accuracy stops improving
    callbacks_list.append(tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_accuracy", factor = 0.1, patience = options.patience, min_delta = 0.0001, min_lr = 1e-5))
  if options.tensorboard:
      # Tensorboard logging
      log_dir = "logs/vgg/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
      tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq = 1, write_grads = True, write_images = True, write_graph = True)
      callbacks_list.append(tensorboard_callback)

  # Prepare batched training and validation datasets
  print("Loading training and validation datasets from %s..." % options.dataset_dir)

  training_data = ImageNet(
    src_dir = options.dataset_dir,
    dataset = "train",
    batch_size = options.batch_size,
    class_filter = class_filter,
    scale_range = scale_range,
    augment = options.augment
  )

  validation_data = ImageNet(
    src_dir = options.dataset_dir,
    dataset = "val",
    batch_size = options.batch_size,
    class_filter = class_filter,
    scale_range = (validation_scale, validation_scale),
    augment = False
  )

  assert training_data.num_classes == validation_data.num_classes
  assert training_data.num_classes == num_classes

  # Perform training
  print("Training model...")
  tic = time.perf_counter()
  training_model.fit(callbacks = callbacks_list, x = training_data.dataset, validation_data = validation_data.dataset, epochs = options.epochs)
  toc = time.perf_counter()
  print("Training time: %1.1f hours" % ((toc - tic) / 3600.0))

  # Save learned model parameters
  if options.save_to is not None:
    model.save_weights(filepath = options.save_to, overwrite = True)
    print("Saved model to %s" % options.save_to)

def evaluate(model, class_filter, scale, options):
  """
  Evaluate the model performance on the validation dataset.
  """
  # Prepare a validation dataset of uncropped and variable-size images, which
  # prevents us from being able to use batching
  print("Loading validation dataset from %s..." % options.dataset_dir)

  evaluation_data = ImageNet(
    src_dir = options.dataset_dir,
    dataset = "val",
    batch_size = 1,
    class_filter = class_filter,
    scale_range = (validation_scale, validation_scale),
    crop_size = None,
    augment = False
  )

  print("Evaluating model on validation data...")
  tic = time.perf_counter()
  metrics = inference_model.evaluate(x = evaluation_data.dataset)
  toc = time.perf_counter()
  print("Results")
  print("-------")
  for i in range(len(inference_model.metrics_names)):
    print("  %s = %1.4f" % (inference_model.metrics_names[i], metrics[i]))
  print("Evaluation time: %1.1f minutes" % ((toc - tic) / 60.0))

def infer(model, url, class_filter, scale, top_k):
  """
  Perform inference on a single image and print the top-K classes.
  """
  print("Loading image from %s..." % url)
  x, original_shape = load_image(url = url, scale = scale, symmetric = False)
  print("Resized image using scale=%d: %dx%d -> %dx%d" % (scale, original_shape[1], original_shape[0], x.shape[2], x.shape[1]))

  print("Running inference...")
  y = model.predict(x)
  y = np.squeeze(y)
  top_k_idx = y.argsort()[-top_k:][::-1]  # get the top-k highest scores in descending order

  idx_to_name = ImageNet.get_index_to_class_name(class_filter = class_filter)
  max_name_width = max([ len(name) for name in idx_to_name.values() ])
  print("%s Score" % "Class".ljust(max_name_width))
  print("%s -----" % "-----".ljust(max_name_width))
  for idx in top_k_idx:
    score = y[idx] * 100.0
    name = idx_to_name[idx]
    print("%s %1.2f%%" % (name.ljust(max_name_width), y[idx] * 100.0))

if __name__ == "__main__":
  parser = argparse.ArgumentParser("VGG16")
  parser.add_argument("--infer", metavar="url", help = "Run inference on single image (file or URL)")
  parser.add_argument("--train", action = "store_true", help = "Train the model on dataset")
  parser.add_argument("--eval", action = "store_true", help = "Evaluate the model on validation dataset using full inference procedure")
  parser.add_argument("--dataset-dir", metavar = "path", type = str, action = "store", help = "Dataset directory")
  parser.add_argument("--scale", metavar = "range", type = utils.int_range_str, action = "store", default = "384", help = "Range of image scales (pixel size of smallest dimension) to randomly sample from")
  parser.add_argument("--no-augment", dest = "augment", action = "store_false", default = True, help = "Disable image augmentation during training (eliminate random horizontal flips and take centered crops only)")
  parser.add_argument("--class-filter", metavar = "names", type = str, action = "store", help = "Restrict to a subset of specified class names")
  parser.add_argument("--list-classes", action = "store_true", help = "List all ImageNet class names")
  parser.add_argument("--batch-size", metavar = "count", type = utils.positive_int, default = 256, action = "store", help = "Training batch size")
  parser.add_argument("--epochs", metavar = "count", type = utils.positive_int, default = "10", action = "store", help = "Number of epochs to train")
  parser.add_argument("--learning-rate", metavar = "value", type = float, default = "0.01", action = "store", help = "Learning rate for SGD and Adam optimizers")
  parser.add_argument("--momentum", metavar = "value", type = float, default = "0.9", action = "store", help = "Momentum factor for SGD optimizer")
  parser.add_argument("--dropout", metavar = "value", type = float, default = "0.5", action = "store", help = "Dropout fraction on last 2 fully connected layers")
  parser.add_argument("--l2", metavar = "value", type = float, default = 2.5e-4, action = "store", help = "L2 regularization on all layers")
  parser.add_argument("--patience", metavar = "value", type = int, action = "store", default = 5, help = "Reduces learning rate by 10 (down to a minimum of 1e-5) when val_acc stops increasing for the given number of epochs")
  parser.add_argument("--top-k", metavar = "value", type = int, default = "5", action = "store", help = "Top-K metric to print")
  parser.add_argument("--log", metavar = "filename", type = str, default = "out.csv", help = "Log metrics to csv file")
  parser.add_argument("--save-to", metavar = "filename", type = str, action = "store", help = "Save final model parameters after training is complete to file")
  parser.add_argument("--load-from", metavar = "filename", type = str, action = "store", help = "Load model parameters from file")
  parser.add_argument("--load-pretrained-weights", action = "store_true", help = "Load Keras pre-trained VGG16 model weights")
  parser.add_argument("--freeze", metavar = "layers", action = "store", help = "Freeze specified layers during training")
  parser.add_argument("--tensorboard", action = "store_true", help = "Emit TensorBoard logging to logs/vgg/")
  options = parser.parse_args()

  if (options.train or options.eval) and not options.dataset_dir:
    parser.error("the following argument is required with --train and --eval: --dataset-dir")
  if options.load_from and options.load_pretrained_weights:
    parser.error("--load-from and --load-pretrained-weights are mutually exclusive")
  if not options.list_classes and not options.infer and not options.train and not options.eval:
    parser.print_help()
    exit()

  # List classes
  if options.list_classes:
    list_classes()

  # Terminate early if no other actions to perform
  if not options.infer and not options.train and not options.eval:
    exit()

  # CUDA available?
  cuda_available = tf.test.is_built_with_cuda()
  gpu_available = tf.test.is_gpu_available(cuda_only = False, min_cuda_compute_capability = None)
  print("CUDA Available: %s" % ("yes" if cuda_available else "no"))
  print("GPU Available : %s" % ("yes" if gpu_available else "no"))

  # Number of classes
  class_filter = construct_class_filter(options)
  num_classes = 1000 if not class_filter else len(class_filter)
  print("Number of classes: %d" % num_classes)

  # Determine image scale to use
  scale_range = (options.scale, options.scale) if len(options.scale) == 1 else options.scale
  validation_scale = int(0.5 * (scale_range[0] + scale_range[1])) # validate at median scale

  # Training and inference models, the latter of which overrides test_step to
  # perform the full inference procedure as described in the VGG-16 paper
  training_model = TrainingModel(
    num_classes = num_classes,
    top_k = options.top_k,
    learning_rate = options.learning_rate,
    momentum = options.momentum,
    dropout = options.dropout,
    l2 = options.l2,
    freeze_layers = options.freeze
  )

  inference_model = InferenceModel(
    num_classes = num_classes,
    top_k = options.top_k,
  )

  # Load weights
  if options.load_pretrained_weights:
    print("Loading Keras pre-trained VGG16 weights...")
    load_pretrained_model_weights(model = training_model)
  elif options.load_from:
    print("Loading Keras model from %s..." % options.load_from)
    training_model.load_weights(filepath = options.load_from, by_name = True)
  inference_model.load_weights(training_model = training_model)

  # Infer
  if options.infer:
    infer(model = inference_model, url = options.infer, class_filter = class_filter, scale = validation_scale, top_k = options.top_k)

  # Train
  if options.train:
    train(model = training_model, class_filter = class_filter, scale_range = scale_range, validation_scale = validation_scale, options = options)

  # Evaluate
  if options.eval:
    evaluate(model = inference_model, class_filter = class_filter, scale = validation_scale, options = options)
