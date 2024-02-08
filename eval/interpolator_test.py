# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""A test script for mid frame interpolation from two input frames.

Usage example:
 python3 -m frame_interpolation.eval.interpolator_test \
   --frame1 <filepath of the first frame> \
   --frame2 <filepath of the second frame> \
   --model_path <The filepath of the TF2 saved model to use>

The output is saved to <the directory of the input frames>/output_frame.png. If
`--output_frame` filepath is provided, it will be used instead.
"""
import os
from typing import Sequence

from . import interpolator as interpolator_lib
from . import util
from absl import app
import numpy as np
from datetime import datetime
import resource
import functools
import natsort
import tensorflow as tf
import gc
from multiprocessing import Process


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


# Controls TF_CCP log level.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def process(start_frame_path, end_frame_path):
  print(str(datetime.now()) + ": Processing with start frame %s and end frame %s" % (start_frame_path, end_frame_path))
  
  # Start the model
  print(str(datetime.now()) + ": Starting interpolator...")
  interpolator = interpolator_lib.Interpolator(
      model_path="pretrained_models/film_net/Style/saved_model",
      align=64,
      block_shape=[1, 1])
  print(str(datetime.now()) + ": Started")

  # Read our new start frame
  print(str(datetime.now()) + ": Reading start frame at " + start_frame_path + "...")
  start_frame = util.read_image(start_frame_path)
  start_frame_batch = np.expand_dims(start_frame, axis=0)
  print(str(datetime.now()) + ": Done")

  # Read our new end frame
  print(str(datetime.now()) + ": Reading end frame at " + end_frame_path + "...")
  end_frame = util.read_image(end_frame_path)
  end_frame_batch = np.expand_dims(end_frame, axis=0)
  print(str(datetime.now()) + ": Done")

  # Batched time.
  print(str(datetime.now()) + ": Batched time(?)...")
  batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
  print(str(datetime.now()) + ": Done")

  # Invoke the model for one mid-frame interpolation.
  print(str(datetime.now()) + ": Invoke the model to get a mid point...")
  mid_frame = interpolator(start_frame_batch, end_frame_batch, batch_dt)[0]
  mid_frame_batch = np.expand_dims(mid_frame, axis=0)
  print(str(datetime.now()) + ": Done")

  # Invoke the model for one quarter-frame interpolation between the start and mid frame
  print(str(datetime.now()) + ": Invoke the model to get a quarter point...")
  quarter_frame = interpolator(start_frame_batch, mid_frame_batch, batch_dt)[0]
  print(str(datetime.now()) + ": Done")

  # Invoke the model for one three-quarter-frame interpolation between the mid and end frame
  print(str(datetime.now()) + ": Invoke the model to get a three-quarter point...")
  three_quarter_frame = interpolator(mid_frame_batch, end_frame_batch, batch_dt)[0]
  print(str(datetime.now()) + ": Done")

  # Write interpolated mid-frame.
  print(str(datetime.now()) + ": Write the files...")
  start_frame_image_name = os.path.basename(start_frame_path).split(".")[0]
  quarter_frame_path = f'xangle/{start_frame_image_name}.25.png'
  util.write_image(quarter_frame_path, quarter_frame)
  mid_frame_path = f'xangle/{start_frame_image_name}.50.png'
  util.write_image(mid_frame_path, mid_frame)
  three_quarter_frame_path = f'xangle/{start_frame_image_name}.75.png'
  util.write_image(three_quarter_frame_path, three_quarter_frame)
  print(str(datetime.now()) + ": Done " + start_frame_image_name)

def _run_interpolator() -> None:
  """Writes interpolated mid frame from a given two input frame filepaths."""

  # Get the list of images
  print(str(datetime.now()) + ": Getting image files...")
  input_frames_list = natsort.natsorted(tf.io.gfile.glob(f'xangle/*.jpg'))
  print(str(datetime.now()) + "Got images: " + str(input_frames_list))

  # Loop through each pair of frames to generate a mid-point image from each
  for index_start in range(0, len(input_frames_list) - 1):
    start_frame_path = input_frames_list[index_start];
    end_frame_path = input_frames_list[index_start + 1];
    
    p = Process(target=process, args=(start_frame_path, end_frame_path,))
    p.start()
    p.join()

  print("Max RAM " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  _run_interpolator()


if __name__ == '__main__':
  app.run(main)
