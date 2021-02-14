#
# VGG16
# dataset.py
# Copyright 2020-2021 Bart Trzynadlowski
#
# Creates a tf.data.Dataset for loading ImageNet images during training and
# evaluation. Only the training and validation datasets are supported because
# the test labels were not available to me.
#
# Classes are assigned (indexed) according to the convention used by the pre-
# trained VGG-16 model included by Keras if the class filter remains empty,
# allowing models to be loaded with Keras pre-trained weights. If a class
# filter is supplied, numbering is computed from the sorted alphabetical
# order of the class names and will not correspond to the Keras convention even
# if all 1000 names are specified explicitly. In fact, that is not possible as
# a couple of classes have redundant names, unfortunately.
#
# Images are pre-processed by subtracting the mean red, green, and blue values
# of the entire dataset from each pixel color component and converted from RGB
# to BGR order. Before mean subtraction, color values range from 0-255.
#
# Data augmentation as described in the VGG-16 paper by Simonyan and Zisserman
# is performed (random horizontal flips, scale jittering, and random crops)
# with the exception of color augmentation, which is not implemented.
#
# The directory structure of the ImageNet dataset directory is expected to be:
#
#   Annotations/
#     CLS-LOC/
#       train/
#         n01440764/
#         ...
#         n15075141/
#       val/
#   Data/
#     CLS-LOC/
#       test/
#       train/
#         n01440764/
#         ...
#         n15075141/
#       val/
#   ImageSets/
#     CLS-LOC/
#

import numpy as np
import os
from pathlib import Path
import tensorflow as tf
import xml.etree.ElementTree as ET

class ImageNet:
  """
  Creates an object with a 'dataset' property containing the tf.data.Dataset.
  The root of the ImageNet data directory is specified by 'src_dir' and
  'dataset' can be 'train' or 'val'. The batch size is controlled by
  'batch_size'. If 'class_filter' is not empty, only the classes specified by
  their names will be included. Images are randomly scaled between the two
  inclusive sizes specified in 'scale_range' and 'crop_size' defines the size
  of the symmetric crop to take. If augmentation is enabled by setting
  'augment', crops are taken randomly rather than from the center and images
  are flipped horizontally with a 50% probability per sample.
  """
  def __init__(self, src_dir, dataset, batch_size, class_filter = {}, scale_range = (224, 224), crop_size = 224, augment = False):
    # Validate parameters
    assert crop_size == None or crop_size <= min(scale_range)
    scale_range = (min(scale_range), max(scale_range))

    # Get image file paths
    all_image_paths = self._get_image_paths(src_dir = src_dir, dataset = dataset)

    # Label each image with a class index [0,num_classes). TensorFlow will
    # compute the one-hot encoding at run-time. It appears that StaticHashTable
    # cannot contain multi-dimensional tensors, hence why we do not store the
    # one-hot encoding there directly.
    self.num_classes, class_index_by_image_path = self._get_labels(image_paths = all_image_paths, src_dir = src_dir, dataset = dataset, class_filter = class_filter)
    image_paths = list(class_index_by_image_path.keys())
    image_class_indexes = list(class_index_by_image_path.values())

    # Construct a tf hashmap of image path -> class index
    tf_class_index_by_image = tf.lookup.StaticHashTable(
      initializer = tf.lookup.KeyValueTensorInitializer(
        keys = tf.constant(image_paths),
        values = tf.constant(image_class_indexes)
      ),
      default_value = tf.constant(-1) # if you are crashing on hash lookups, it is likely that you passed in an invalid directory and are hitting this
    )

    # Construct tf.data.Dataset pipeline that converts image paths ->
    # shuffled, preprocessed images
    paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    paths_ds = paths_ds.shuffle(buffer_size = len(image_paths), reshuffle_each_iteration = True)  # reshuffle each epoch
    image_ds = paths_ds.map(
      lambda path: (self._load_image(path, scale_range, crop_size, augment), self._get_onehot_label(path, tf_class_index_by_image, self.num_classes)),
      num_parallel_calls = tf.data.experimental.AUTOTUNE
    )
    image_ds = image_ds.batch(batch_size)
    image_ds = image_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    self.dataset = image_ds

  @staticmethod
  def _get_onehot_label(image_path, tf_class_index_by_image_path, num_classes):
    """
    Converts an image path (string) into a one-hot label (tensor) using the
    supplied map.
    """
    return tf.one_hot(tf_class_index_by_image_path.lookup(image_path), depth = num_classes)

  @staticmethod
  def _load_image(image_path, scale_range, crop_size, augment):
    """
    Loads an image using TensorFlow. The image is scaled, cropped, and pre-
    processed. If 'augment' is true, the crop is taken from a random location
    rather than the center and a horizontal flip is randomly applied.
    """
    # Select scale (randomly if a range was specified)
    if scale_range[0] != scale_range[1]:
      scale = tf.random.uniform(shape = [], minval = scale_range[0], maxval = (scale_range[1]+1), dtype = tf.int32) # samples [minval,maxval), hence the +1
    else:
      scale = scale_range[0]

    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels = 3) # uint8 in range [0,MAX]

    # Compute new size for the image, proportionally, such that the minimum
    # dimension is scale
    dims = tf.shape(image)  # height, width, channels
    scale_op = tf.cast(scale, dtype = tf.float64)
    new_dims = tf.cond(dims[0] > dims[1], lambda: [ int((dims[0] / dims[1]) * scale_op), scale ], lambda: [ scale, int((dims[1] / dims[0]) * scale_op) ])

    # Resize the image and, optionally, crop
    image = tf.image.convert_image_dtype(image, dtype = tf.float32) # convert to float in range [0,1)
    image = tf.image.resize(image, new_dims)
    if crop_size != None:
      if augment:
        # Augmentation: Random crop
        image = tf.image.random_crop(image, size = [ crop_size, crop_size, 3 ])
      else:
        # No augmentation: Center crop
        crop_offset_width = int((new_dims[1] - crop_size) / 2)
        crop_offset_height = int((new_dims[0] - crop_size) / 2)
        image = tf.image.crop_to_bounding_box(image, offset_height = crop_offset_height, offset_width = crop_offset_width, target_height = crop_size, target_width = crop_size)

    # Preprocess: convert to appropriate floating point representation
    image = tf.image.convert_image_dtype(image, dtype = tf.uint8)   # [0,1) -> [0,255]
    image = tf.cast(image, dtype = tf.float32)                      # [0,255] -> [0.0,255.0] because preprocess_input() on uint8 data only performs RGB->BGR step but does not subtract mean
    image = tf.keras.applications.vgg16.preprocess_input(x = image) # RGB->BGR, each channel centered about ImageNet mean, no scaling

    # Augmentation
    if augment:
      image = tf.image.random_flip_left_right(image)
    return image

  @staticmethod
  def _get_image_paths(src_dir, dataset):
    """
    Returns a list of all image paths for the given dataset.
    """
    dataset_path = os.path.join(src_dir, "Data", "CLS-LOC", dataset)
    if dataset == "train":
      return [ str(path) for path in Path(dataset_path).glob("*/*.JPEG") ]
    elif dataset == "val":
      return  [ str(path) for path in Path(dataset_path).glob("*.JPEG") ]
    else:
      raise Exception("Invalid dataset. Must be 'train' or 'val'.")

  @staticmethod
  def _get_class_id_to_index(class_filter):
    """
    Returns a mapping of class ID to label index according, where index is
    [0,N-1] where N=1000 if the class filter is empty (all classes included),
    and N<1000 otherwise. Exclusively in the case of an empty class filter, the
    indices match the numbering used by Keras in its VGG-16 model output layer,
    allowing Keras pre-trained weights to be loaded into the model.
    """
    classes = tf.keras.applications.imagenet_utils.decode_predictions(np.eye(1000), top = 1)
    id_to_idx = { classes[idx][0][0]: idx for idx in range(len(classes)) }

    if len(class_filter) > 0:
      _ = ImageNet.get_index_to_class_name(class_filter = class_filter) # handles validation for us
      class_filter = sorted(class_filter) # must sort for consistent index ordering
      name_to_id = { classes[idx][0][1]: classes[idx][0][0] for idx in range(len(classes)) }
      id_to_idx = { name_to_id[class_filter[idx]]: idx for idx in range(len(class_filter)) }

    return id_to_idx

  @staticmethod
  def get_index_to_class_name(class_filter = {}):
    """
    Returns a mapping of class index to (human-readable) class name. This
    mapping is *not* invertible because ImageNet has more than one class with
    the exact same name. The class indices range from [0,N] where N=1000 in the
    case of an empty class filter (all classes included) and correspond to the
    Keras VGG-16 model convention. Otherwise, N<1000.
    """
    classes = tf.keras.applications.imagenet_utils.decode_predictions(np.eye(1000), top = 1)
    idx_to_name = { idx: classes[idx][0][1] for idx in range(len(classes)) }

    # If filter specificed, re-number the indices to be contiguous from 0 and
    # packed
    if len(class_filter) > 0:
      # Sort for consistent order and validate class names in filter
      class_filter = sorted(class_filter) # must sort for consistent index ordering
      class_names = idx_to_name.values()
      valid_names = set([ name for name in class_filter if name in class_names ])
      invalid_names = set(class_filter) - valid_names
      if len(invalid_names) > 0:
        raise Exception("Invalid class names specified in class filter: %s" % str(",".join(invalid_names)))

      # Re-number
      idx_to_name = { idx: class_filter[idx] for idx in range(len(class_filter)) }

    return idx_to_name

  @staticmethod
  def _get_class_id_from_annotation_xml(xml_path):
    tree = ET.parse(xml_path)
    assert tree != None, "Failed to parse %s" % xml_path
    root = tree.getroot()

    # ImageNet images may contain multiple objects but they must all be of the
    # same class
    classes = set()
    for obj in root.findall("object"):
      assert len(obj.findall("name")) == 1
      name = obj.find("name").text
      classes.add(name)
    assert len(classes) == 1, "Expected %s to have one class but found %d" % (xml_path, len(classes))
    return list(classes)[0]

  @staticmethod
  def _get_class_id_by_val_file(image_paths, src_dir, valid_class_ids):
    # Get all annotation XML files
    annotation_root_path = os.path.join(src_dir, "Annotations", "CLS-LOC", "val")
    annotation_paths = [ path for path in Path(annotation_root_path).glob("ILSVRC2012_val_*.xml") ]
    assert len(annotation_paths) == len(image_paths), "Number of val images does not match the number of annotation files. Is your ImageNet dataset complete?"

    # For each val image, get its class id from the annotation file
    id_by_file = {}
    for image_path in image_paths:
      filename = os.path.basename(image_path)
      title = os.path.splitext(filename)[0]
      xml_filename = title + ".xml"
      xml_path = os.path.join(src_dir, "Annotations", "CLS-LOC", "val", xml_filename)
      id = ImageNet._get_class_id_from_annotation_xml(xml_path)
      if id in valid_class_ids:
        id_by_file[image_path] = id
    return id_by_file

  @staticmethod
  def _get_class_ids_from_image_paths(image_paths, src_dir, dataset, class_id_to_idx):
    """
    Given a series of images in directories named after their class ID, returns
    a mapping of image path to class ID.
    """
    if dataset == "train":
      return { image_path: os.path.split(os.path.dirname(image_path))[-1] for image_path in image_paths }
    elif dataset == "val":
      return ImageNet._get_class_id_by_val_file(image_paths = image_paths, src_dir = src_dir, valid_class_ids = set(class_id_to_idx.keys()))
    else:
      raise Exception("Invalid dataset. Must be 'train' or 'val'.")

  @staticmethod
  def _get_labels(image_paths, src_dir, dataset, class_filter):
    """
    Returns the number of classes and the class index for each image.
    """
    # Mapping of class IDs to the class index (filtered)
    class_id_to_idx = ImageNet._get_class_id_to_index(class_filter = class_filter)

    # Get class IDs from the directory structure (coimpletely unfiltered)
    class_id_by_image_path = ImageNet._get_class_ids_from_image_paths(image_paths = image_paths, src_dir = src_dir, dataset = dataset, class_id_to_idx = class_id_to_idx)

    # Remove images belonging to any classes that may have been filtered out by
    # the class filter
    class_id_by_image_path = { path: class_id_by_image_path[path] for path in class_id_by_image_path if class_id_by_image_path[path] in class_id_to_idx }

    # Each class should be represented in the image set
    num_classes = len(set(class_id_by_image_path.values()))
    assert num_classes == len(class_id_to_idx)

    onehot_by_id = { id: tf.keras.utils.to_categorical(idx, num_classes) for id, idx in class_id_to_idx.items() }
    class_index_by_image_path = { image_path: class_id_to_idx[id] for image_path, id in class_id_by_image_path.items() }
    return num_classes, class_index_by_image_path