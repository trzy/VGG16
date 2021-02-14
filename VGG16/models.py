#
# VGG16
# models.py
# Copyright 2020-2021 Bart Trzynadlowski
#
# Defines two versions of the VGG-16 model: one for training that accepts a
# fixed-size image and outputs a single result vector, and another for
# inference that accepts a variable-sized image and outputs a 2D map of
# result vectors.
#
# Reference paper:
#
#   Very Deep Convolutional Networks for Large-Scale Image Recognition
#   Karen Simonyan, Andrew Zisserman
#   ICLR 2015
#

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.optimizers import SGD

class TrainingModel(Sequential):
  """
  Build a trainable model with the desired number of output classes and hyper-
  parameters. Top-K accuracy is used as a metric, where the value of k is
  'top_k'. Layers may be frozen during training, to support transfer learning,
  by passing a comma-separated string of layer names in 'freeze_layers'.

  L2 regularization can be applied to all layers, including convolutional
  layers, as was originally done by Simonyan and Zisserman. However, the
  value given for weight decay in the paper must be divided by 2 in order to be
  converted to an equivalent L2 penalty because Keras adds the L2 penalty to
  the loss and then differentiated with respect to the weights (introducing a
  factor of 2 that must be canceled). See:
  https://bbabenko.github.io/weight-decay/

  Validation steps during training use this model, which is in contrast to the
  paper, which uses a more thorough inference procedure implemented by the
  inference model in this module.
  """
  def __init__(self, num_classes, top_k, learning_rate, momentum, dropout, l2, freeze_layers):
    super().__init__()
    self.num_classes = num_classes

    regularizer = None if l2 == None else tf.keras.regularizers.l2(l2)
    initial_weights = glorot_normal()

    self.add( Conv2D(name = "block1_conv1", input_shape = (224,224,3), kernel_size = (3,3), strides = 1, filters = 64, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
    self.add( Conv2D(name = "block1_conv2", kernel_size = (3,3), strides = 1, filters = 64, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
    self.add( MaxPooling2D(pool_size = 2, strides = 2) )

    self.add( Conv2D(name = "block2_conv1", kernel_size = (3,3), strides = 1, filters = 128, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
    self.add( Conv2D(name = "block2_conv2", kernel_size = (3,3), strides = 1, filters = 128, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
    self.add( MaxPooling2D(pool_size = 2, strides = 2) )

    self.add( Conv2D(name = "block3_conv1", kernel_size = (3,3), strides = 1, filters = 256, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
    self.add( Conv2D(name = "block3_conv2", kernel_size = (3,3), strides = 1, filters = 256, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
    self.add( Conv2D(name = "block3_conv3", kernel_size = (3,3), strides = 1, filters = 256, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
    self.add( MaxPooling2D(pool_size = 2, strides = 2) )

    self.add( Conv2D(name = "block4_conv1", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
    self.add( Conv2D(name = "block4_conv2", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
    self.add( Conv2D(name = "block4_conv3", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
    self.add( MaxPooling2D(pool_size = 2, strides = 2) )

    self.add( Conv2D(name = "block5_conv1", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
    self.add( Conv2D(name = "block5_conv2", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
    self.add( Conv2D(name = "block5_conv3", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
    self.add( MaxPooling2D(pool_size = 2, strides = 2) )

    self.add( Flatten() )

    self.add( Dense(name = "fc1", units = 4096, activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
    if dropout > 0:
      self.add( Dropout(dropout) )
    self.add( Dense(name = "fc2", units = 4096, activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer) )
    if dropout > 0:
      self.add( Dropout(dropout) )
    self.add( Dense(name = "predictions", units = num_classes, kernel_initializer = initial_weights) )
    self.add( Activation(activation = "softmax") ) # one-hot classification

    optimizer = SGD(learning_rate = learning_rate, momentum = momentum)
    self.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k = top_k)])

    self._freeze_layers(layers = freeze_layers)

  def _freeze_layers(self, layers):
    frozen_layers = [] if layers == None else layers.split(",")
    for layer in self.layers:
      if layer.name in frozen_layers:
        layer.trainable = False

class InferenceModel(Sequential):
  """
  Build a model suitable for inference. Both the evaluation and prediction
  methods are overridden to perform the inference procedure described in the
  paper.
  """
  def __init__(self, num_classes, top_k):
    super().__init__()
    self.num_classes = num_classes

    self.add( Conv2D(name = "block1_conv1", input_shape = (None,None,3), kernel_size = (3,3), strides = 1, filters = 64, padding = "same", activation = "relu") )
    self.add( Conv2D(name = "block1_conv2", kernel_size = (3,3), strides = 1, filters = 64, padding = "same", activation = "relu") )
    self.add( MaxPooling2D(pool_size = 2, strides = 2) )

    self.add( Conv2D(name = "block2_conv1", kernel_size = (3,3), strides = 1, filters = 128, padding = "same", activation = "relu") )
    self.add( Conv2D(name = "block2_conv2", kernel_size = (3,3), strides = 1, filters = 128, padding = "same", activation = "relu") )
    self.add( MaxPooling2D(pool_size = 2, strides = 2) )

    self.add( Conv2D(name = "block3_conv1", kernel_size = (3,3), strides = 1, filters = 256, padding = "same", activation = "relu") )
    self.add( Conv2D(name = "block3_conv2", kernel_size = (3,3), strides = 1, filters = 256, padding = "same", activation = "relu") )
    self.add( Conv2D(name = "block3_conv3", kernel_size = (3,3), strides = 1, filters = 256, padding = "same", activation = "relu") )
    self.add( MaxPooling2D(pool_size = 2, strides = 2) )

    self.add( Conv2D(name = "block4_conv1", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu") )
    self.add( Conv2D(name = "block4_conv2", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu") )
    self.add( Conv2D(name = "block4_conv3", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu") )
    self.add( MaxPooling2D(pool_size = 2, strides = 2) )

    self.add( Conv2D(name = "block5_conv1", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu") )
    self.add( Conv2D(name = "block5_conv2", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu") )
    self.add( Conv2D(name = "block5_conv3", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu") )
    self.add( MaxPooling2D(pool_size = 2, strides = 2) )

    # Names are misnomers as these are no longer fully-connected layers but
    # rather convolutional layers that take the same weights as their fully-
    # connected counterparts. They create a map of solutions, as from the
    # original training network, but applied over arbitrary-sized input images.
    # The original 4096-unit layers can be thought of as having been pivoted
    # into the third dimension -- the number of filters of the convolutions.
    self.add( Conv2D(name = "fc1", kernel_size = (7,7), strides = 1, filters = 4096, activation = "relu") )
    self.add( Conv2D(name = "fc2", kernel_size = (1,1), strides = 1, filters = 4096, activation = "relu") )
    self.add( Conv2D(name = "predictions", kernel_size = (1,1), strides = 1, filters = num_classes, activation = "softmax") )

    self.add( GlobalAveragePooling2D() )  # (1, height, width, num_classes) -> (1, num_classes)

    optimizer = SGD() # params don't matter as we will never train with this
    self.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k = top_k)])

  def test_step(self, data):
    """
    Override Keras 'test_step' to use custom logic consistent with paper.
    """
    # Similar to customizing train_step: https://keras.io/guides/customizing_what_happens_in_fit/
    # Source code for the Model superclass (and its test_step method): https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/training.py
    assert len(data) == 2 # we don't support sample weights
    x, y = data
    y_predicted = self._custom_predict(x = x)
    self.compiled_loss(y, y_predicted, regularization_losses=self.losses)
    self.compiled_metrics.update_state(y, y_predicted)
    return { metric.name: metric.result() for metric in self.metrics }

  def predict_step(self, data):
    """
    Override Keras 'predict_step' to use custom logic consistent with paper.
    """
    return self._custom_predict(x = data)

  def _custom_predict(self, x):
    # The VGG testing procedure requires augmentation, which evaluates a
    # flipped version of the image and averages the predictions with those of
    # the original.
    x_flipped = tf.image.flip_left_right(x)
    y_predicted_original = self(x, training = False)
    y_predicted_flipped = self(x_flipped, training = False)
    y_predicted = 0.5 * (y_predicted_original + y_predicted_flipped)
    return y_predicted

  def load_weights(self, training_model):
    """
    Loads weights from a trained model. The last few layers of the inference
    model differ from the training model in that they are structured as
    convolutions and produce a map of outputs but the filters take the same
    weights and in the same order as the training model, albeit requiring that
    they be re-shaped, which this method handles.
    """
    assert training_model.num_classes == self.num_classes

    #
    # Training network final layer weight and bias shapes:
    #
    #   fc1         (25088, 4096), (4096,)
    #   fc2         (4096, 4096), (4096,)
    #   predictions (4096, num_classes), (num_classes,)
    #
    # Inference network final layer weight and bias shapes:
    #
    #   fc1         (7, 7, 512, 4096), (4096,)
    #   fc2         (1, 1, 4096, 4096), (4096,)
    #   predictions (1, 1, 4096, num_classes), (num_classes,)
    #
    # Trained weights must be reshaped to their convolutional equivalents.
    #

    map_shapes = {
      "fc1": (7, 7, 512, 4096),
      "fc2": (1, 1, 4096, 4096),
      "predictions": (1, 1, 4096, training_model.num_classes)
    }

    for trained_layer in training_model.layers:
      name = trained_layer.name
      trained_weights = trained_layer.get_weights()
      if len(trained_weights) > 0:
        if trained_layer.name in map_shapes:
          # Need to reshape the weights for the final, dense layers
          layer = [ layer for layer in self.layers if layer.name == name ][0]
          weights = trained_weights[0]
          biases = trained_weights[1]
          new_shape = map_shapes[name]
          new_weights = [ weights.reshape(new_shape), biases ]
          layer.set_weights(new_weights)
        else:
          # Layers are identical in both models, simply copy over weights
          layer = [ layer for layer in self.layers if layer.name == name ][0]
          layer.set_weights(trained_weights)