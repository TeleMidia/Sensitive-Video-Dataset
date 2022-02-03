# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Produces tfrecord files similar to the YouTube-8M dataset.

It processes a CSV file containing lines like "<input_video_path>,<labels>", where
<input_video_path> must be a path of a video, and <labels> must be an integer list
joined with semi-colon ";". It processes all videos and outputs tfrecord file
onto --output_tfrecords_dir.

It assumes that you have OpenCV installed and properly linked with ffmpeg (i.e.
function `cv2.VideoCapture().open('/path/to/some/video')` should return True).

The binary only processes the video stream (images) and not the audio stream.
"""
import os
import sys

import cv2
import feature_extractor_class_image
import numpy as np
import tensorflow as tf
from tensorflow import app
from tensorflow import flags
from tqdm import tqdm
from numpy import genfromtxt

# In OpenCV3.X, this is available as cv2.CAP_PROP_POS_MSEC
# In OpenCV2.X, this is available as cv2.cv.CV_CAP_PROP_POS_MSEC
CAP_PROP_POS_MSEC = 0


  # DEFINE_string(
  #     'output_tfrecords_dir', None,
  #     'File containing tfrecords will be written at this path.')
  # DEFINE_string(
  #     'input_videos_csv', None,
  #     'CSV file with lines "<input_video_path>,<labels>", where '
  #     '<input_video_path> must be a path of a video and <labels> '
  #     'must be an integer list joined with semi-colon ";"')
  # # Optional 
  # DEFINE_string('model_dir', os.path.join(os.getenv('HOME'), 'yt8m'),
  #                     'Directory to store model files. It defaults to ~/yt8m')

  # # The following are set to match the YouTube-8M dataset format.
  # DEFINE_integer('frames_per_second', 1,
  #                      'This many frames per second will be processed')
  # DEFINE_boolean(
  #     'skip_frame_level_features', False,
  #     'If set, frame-level features will not be written: only '
  #     'video-level features will be written with feature '
  #     'names mean_*')
  # DEFINE_string(
  #     'labels_feature_key', 'labels',
  #     'Labels will be written to context feature with this '
  #     'key, as int64 list feature.')
  # DEFINE_string(
  #     'image_feature_key', 'rgb',
  #     'Image features will be written to sequence feature with '
  #     'this key, as bytes list feature, with only one entry, '
  #     'containing quantized feature string.')
  # DEFINE_string(
  #     'input_video_path_feature_key', 'id',
  #     'Input <input_video_path> will be written to context feature '
  #     'with this key, as bytes list feature, with only one '
  #     'entry, containing the file path of the video. This '
  #     'can be used for debugging but not for training or eval.')
  # DEFINE_boolean(
  #     'insert_zero_image_features', False,
  #     'If set, inserts features with name "audio" to be 128-D '
  #     'zero vectors. This allows you to use YouTube-8M '
  #     'pre-trained model.')
def get_video_duration(video_capture):
  fps = video_capture.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
  frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
  duration = frame_count/fps

  #print('duration (S) = ' + str(duration))
  #minutes = int(duration/60)
  #seconds = duration%60
  #print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))
  return duration


def frame_iterator(filename, every_ms=1000, max_num_frames=300):
  """Uses OpenCV to iterate over all frames of filename at a given frequency.

  Args:
    filename: Path to video file (e.g. mp4)
    every_ms: The duration (in milliseconds) to skip between frames.
    max_num_frames: Maximum number of frames to process, taken from the
      beginning of the video.

  Yields:
    RGB frame with shape (image height, image width, channels)
  """
  video_capture = cv2.VideoCapture()
  
  if not video_capture.open(filename):
    #aise Exception('Cannot open video file ' + filename)
    #print('Error: Cannot open video file ' + filename)
    return
  last_ts = -99999  # The timestamp of last retrieved frame.
  num_retrieved = 0

  #update based on framerate
  #fps = video_capture.get(cv2.CAP_PROP_FPS)  
  #every_ms = every_ms/fps
  #print('Video ms:', every_ms)
  #print(get_video_duration(video_capture))
  while num_retrieved < max_num_frames:
    # Skip frames
    while video_capture.get(CAP_PROP_POS_MSEC) < every_ms + last_ts:
      if not video_capture.read()[0]:
        return
        #raise Exception('Error reading video frame.')

    last_ts = video_capture.get(CAP_PROP_POS_MSEC)
    has_frames, frame = video_capture.read()
    if not has_frames:
      break
    yield frame
    num_retrieved += 1


def _int64_list_feature(int64_list):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=int64_list))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _make_bytes(int_array):
  if bytes == str:  # Python2
    return ''.join(map(chr, int_array))
  else:
    return bytes(int_array)


def quantize(features, min_quantized_value=-2.0, max_quantized_value=2.0):
  """Quantizes float32 `features` into string."""
  assert features.dtype == 'float32'
  assert len(features.shape) == 1  # 1-D array
  features = np.clip(features, min_quantized_value, max_quantized_value)
  quantize_range = max_quantized_value - min_quantized_value
  features = (features - min_quantized_value) * (255.0 / quantize_range)
  #features = [int(round(f)) for f in features]
  features = [round(f).astype(np.uint8) for f in features]

  return features#_make_bytes(


def extract_image_features(
  input_video_path,
  model_dir=os.path.join(os.getenv('HOME'), 'yt8m'),
  which_gpu=0,
  frames_per_second=1, 
  csv_file=None,
  skip_frame_level_features=False, 
  labels_feature_key='labels', 
  image_feature_key='rgb', 
  input_video_path_feature_key='id',
  insert_zero_image_features=False
  ):
  extractor = feature_extractor_class_image.YouTube8MFeatureExtractor(model_dir=model_dir,which_gpu=which_gpu)
  rgb_features = []
  sum_rgb_features = None
  ms_interval = 960.0
  #print(f'every_frame:{ms_interval}, frames per second: {frames_per_second}, total: {ms_interval / frames_per_second}')
  frames = frame_iterator(input_video_path, every_ms=(ms_interval),max_num_frames=100000000)  # HERE WE DEFINE THE RATE OF FEATURE EXTRACTION #IN THE VGGISH GIT, IT SAYS THAT EACH MEL IS GROUPED BY 0.96s WINDOWS
  for rgb in frames:

    features = extractor.extract_rgb_frame_features(rgb[:, :, ::-1])
    if sum_rgb_features is None:
      sum_rgb_features = features
    else:
      sum_rgb_features += features
    rgb_features.append(features)#rgb_features.append(_bytes_feature(quantize(features)))

  if not rgb_features:
    raise Exception('Cannot open video file: ' + input_video_path)

  #print('Extracted image features with success!')
  return rgb_features