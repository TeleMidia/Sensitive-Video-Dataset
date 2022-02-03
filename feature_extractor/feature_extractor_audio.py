# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""A simple demonstration of running VGGish in inference mode.

This is intended as a toy example that demonstrates how the various building
blocks (feature extraction, model definition and loading, postprocessing) work
together in an inference context.

A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted
into log mel spectrogram examples, fed into VGGish, the raw embedding output is
whitened and quantized, and the postprocessed embeddings are optionally written
in a SequenceExample to a TFRecord file (using the same format as the embedding
features released in AudioSet).

Usage:
	# Run a WAV file through the model and print the embeddings. The model
	# checkpoint is loaded from vggish_model.ckpt and the PCA parameters are
	# loaded from vggish_pca_params.npz in the current directory.
	$ python vggish_inference_demo.py --wav_file /path/to/a/wav/file

	# Run a WAV file through the model and also write the embeddings to
	# a TFRecord file. The model checkpoint and PCA parameters are explicitly
	# passed in as well.
	$ python vggish_inference_demo.py --wav_file /path/to/a/wav/file \
																		--tfrecord_file /path/to/tfrecord/file \
																		--checkpoint /path/to/model/checkpoint \
																		--pca_params /path/to/pca/params

	# Run a built-in input (a sine wav) through the model and print the
	# embeddings. Associated model files are read from the current directory.
	$ python vggish_inference_demo.py
"""

from __future__ import print_function

import numpy as np
import six
import tensorflow as tf
import csv
import os
from numpy import genfromtxt

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

# DEFINE_string(
#     'wav_file', None,
#     'Path to a wav file. Should contain signed 16-bit PCM samples. '
#     'If none is provided, a synthetic sound is used.')

# DEFINE_string(
#     'checkpoint', 'vggish_model.ckpt',
#     'Path to the VGGish checkpoint file.')

# DEFINE_string(
#     'pca_params', 'vggish_pca_params.npz',
#     'Path to the VGGish PCA parameters file.')

# DEFINE_string(
#     'tfrecord_file', None,
#     'Path to a TFRecord file where embeddings will be written.')

# DEFINE_string(
#     'csv_file', None,
#     'Path to a csv file where embeddings will be written.')

def save(directory, filename, saver, sess):
	# Add ops to save and restore all the variables.
	if not os.path.exists(directory):
		os.makedirs(directory)
	filepath = os.path.join(directory, filename)
	saver.save(sess, filepath)
	return filepath

from tensorflow.python.tools import freeze_graph
import pickle
def save_as_pb(directory, filename, sess):
	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()
	if not os.path.exists(directory):
		os.makedirs(directory)

	# Save check point for graph frozen later
	ckpt_filepath = save(directory=directory, filename=filename, saver=saver, sess=sess)
	pbtxt_filename = filename + '.pbtxt'
	pbtxt_filepath = os.path.join(directory, pbtxt_filename)
	pb_filepath = os.path.join(directory, filename + '.pb')
	# This will only save the graph but the variables will not be saved.
	# You have to freeze your model first.
	tf.train.write_graph(graph_or_graph_def=sess.graph_def, logdir=directory, name=pbtxt_filename, as_text=True)

	# Freeze graph
	freeze_graph.freeze_graph(input_graph=pbtxt_filepath, input_saver='', input_binary=False, input_checkpoint=ckpt_filepath, output_node_names=vggish_params.OUTPUT_OP_NAME, restore_op_name='save/restore_all', filename_tensor_name='save/Const:0', output_graph=pb_filepath, clear_devices=True, initializer_nodes='')
	
	return pb_filepath

def load_pb(model_filepath):
	'''
	Load trained model.
	'''
	graph = tf.Graph()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.InteractiveSession(graph = graph)

	with tf.gfile.GFile(model_filepath, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	# Define input tensor
	#input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 1152], name='mlp/inputX')
	tf.import_graph_def(graph_def,name='')#, {'import/mlp/inputX': input_tensor}

	return sess, graph

def extract_audio_features(checkpoint, pca_params,wav_file, which_gpu = 0, tfrecord_file=None, csv_file=None, loaded_file = None):
	# In this simple example, we run the examples from a single audio file through
	# the model. If none is provided, we generate a synthetic input.
	wav_file = str(wav_file)#.replace('-','\-')
	pproc = vggish_postprocess.Postprocessor(pca_params)
	examples_batch = vggish_input.wavfile_to_examples(wav_file)

	with open('input_tf1.pickle', 'wb') as handle:
		pickle.dump(examples_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)


	# Assume that you have 8GB of GPU memory and want to allocate ~4GB:
	#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.5)
	#config = tf.ConfigProto(gpu_options=gpu_options)#, device_count = {'GPU': which_gpu}

	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	with tf.Graph().as_default(), tf.Session(config=config) as sess:
		# Define the model in inference mode, load the checkpoint, and
		# locate input and output tensors.
		vggish_slim.define_vggish_slim(training=False)
		vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint)
		# sess, graph = load_pb('./vggish_slim_tf1.pb') 
		features_tensor = sess.graph.get_tensor_by_name(
				vggish_params.INPUT_TENSOR_NAME)
		embedding_tensor = sess.graph.get_tensor_by_name(
				vggish_params.OUTPUT_TENSOR_NAME)

		# Run inference and postprocessing.
		[embedding_batch] = sess.run([embedding_tensor],feed_dict={features_tensor: examples_batch})
		#print(embedding_batch)

		with open('output_tf1.pickle', 'wb') as handle:
			pickle.dump(embedding_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)
		# print(len(embedding_batch))
		audio_features = pproc.postprocess(embedding_batch) # I COMMENTED THE QUANTIZATION STEP IN THE POST PROCESSING CLASS
		audio_features = np.array(audio_features)

		with open('output_tf1_2.pickle', 'wb') as handle:
			pickle.dump(embedding_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)

		#audio_features = np.average(audio_features, axis=0)
		# print(save_as_pb('./', 'vggish_slim_tf1', sess))

		return audio_features
