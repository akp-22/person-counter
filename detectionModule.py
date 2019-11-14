import numpy as np
import os
import six
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import sys

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import pdb
import cv2

sys.path.append("..")
from object_detection.utils import ops as utils_ops

from utils import label_map_util
from utils import visualization_utils as vis_util


class DetectionModule():
	
	def __init__(self, label, threshold_conf):
		self.object_to_detect = label
		self.min_score_thresh = threshold_conf
	
	def load_model(self):
		# What model to download.
		MODEL_NAME = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
		MODEL_FILE = MODEL_NAME + '.tar.gz'
		DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

		# path to the object detection model.
		PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

		# List of the strings that is used to add correct label for each box.
		PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
		self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

		print("Downloading weights started \n")	
		tar_file = tarfile.open(MODEL_FILE)
		for file in tar_file.getmembers():
			file_name = os.path.basename(file.name)
			if 'frozen_inference_graph.pb' in file_name:
				tar_file.extract(file, os.getcwd())

		print("Downloading weights done \n")		
		self.detection_graph = tf.Graph()
		with self.detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')
	
	def run_inference_for_single_image(self, image, graph):
		with graph.as_default():
			with tf.Session() as sess:
				# Get handles to input and output tensors
				ops = tf.get_default_graph().get_operations()
				all_tensor_names = {output.name for op in ops for output in op.outputs}
				tensor_dict = {}
				for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
					tensor_name = key + ':0'
					if tensor_name in all_tensor_names:
						tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
						  tensor_name)
				if 'detection_masks' in tensor_dict:
					# The following processing is only for single image
					detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
					detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
					# Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
					real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
					detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
					detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
					detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
						detection_masks, detection_boxes, image.shape[1], image.shape[2])
					detection_masks_reframed = tf.cast(
						tf.greater(detection_masks_reframed, 0.5), tf.uint8)
					# Follow the convention by adding back the batch dimension
					tensor_dict['detection_masks'] = tf.expand_dims(
						detection_masks_reframed, 0)
				image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

				# Run inference
				output_dict = sess.run(tensor_dict,
									 feed_dict={image_tensor: image})

				# all outputs are float32 numpy arrays, so convert types as appropriate
				output_dict['num_detections'] = int(output_dict['num_detections'][0])
				output_dict['detection_classes'] = output_dict[
				  'detection_classes'][0].astype(np.int64)
				output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
				output_dict['detection_scores'] = output_dict['detection_scores'][0]
				if 'detection_masks' in output_dict:
					output_dict['detection_masks'] = output_dict['detection_masks'][0]
		return output_dict
	
	def return_valid_boxes( self,
							image,
							boxes,
							classes,
							scores,
							category_index,
							use_normalized_coordinates=True,
							max_boxes_to_draw=20):

		Boxes = []
		max_boxes_to_draw = boxes.shape[0]
		for i in range(min(max_boxes_to_draw, boxes.shape[0])):
			if scores is None or scores[i] > self.min_score_thresh:
				box = tuple(boxes[i].tolist())
				if classes[i] in six.viewkeys(category_index):
					class_name = category_index[classes[i]]['name']
				else:
					class_name = 'N/A'
				if not class_name == self.object_to_detect:
					continue
				else:
					if use_normalized_coordinates:
						ymin, xmin, ymax, xmax = box
						im_width, im_height, _ = np.shape(image)
						bounding_box = np.array([xmin * im_height, ymin * im_width, xmax * im_height,
													   ymax * im_width])
						Boxes.append(bounding_box)
		return Boxes				
		
	def inference(self, image):
				
		image_np = image
		# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		image_np_expanded = np.expand_dims(image_np, axis=0)
		# Actual detection
		output_dict = self.run_inference_for_single_image(image_np_expanded, self.detection_graph)
		# Retaining valid boxes detection.
		return self.return_valid_boxes(image_np, output_dict['detection_boxes'],
								output_dict['detection_classes'], output_dict['detection_scores'], 
								self.category_index)	