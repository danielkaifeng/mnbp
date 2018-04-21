#coding=utf8
import os
import tensorflow as tf
from sys import argv

import json

def create_record(x_path, y_path, json_path):
	train_writer = tf.python_io.TFRecordWriter("./data/train.tfrecords")
	test_writer = tf.python_io.TFRecordWriter("./data/test.tfrecords")


	anno_json = json.load(open(json_path, 'r'))

	f1 = open(x_path, 'r')
	f2 = open(y_path, 'r')

	import random
	i = 1

	while True:
			xline = f1.readline()
			yline = f2.readline()
			if xline == "" and yline == "":
					break

			lx = xline.strip().split(',')
			ly = yline.strip().split()
	
			ID = lx[0]
			assert ID == ly[0] 

			data = [float(x) for x in lx[1:]]
			y = [float(x) for x in ly[1:]]

			binary_label = convert_label_to_multihot(anno_json[ID],299)
			data += binary_label

			train = True
			if random.randint(0,9) == 1:
					train = False


			print str(i) + '\t' + ID
			i += 1
		

			example = tf.train.Example(features=tf.train.Features(feature={
				"x": tf.train.Feature(float_list=tf.train.FloatList(value=data)),
				'y': tf.train.Feature(float_list=tf.train.FloatList(value=y))
			}))

			if train:
				train_writer.write(example.SerializeToString())
			else:
				test_writer.write(example.SerializeToString())


	train_writer.close()
	test_writer.close()

def convert_label_to_multihot(labels, num_class):
	# label with 7 class example: [3, 5]  ==>  [0,0,0,1,0,1,0]
	multihot_label = [0] * num_class
	for i in labels:
		multihot_label[int(i)] = 1

	return multihot_label



def read_and_decode(filename):
	print 'read and decode data...'
	filename_queue = tf.train.string_input_producer([filename])

	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example,
									   features={
										   'x': tf.FixedLenFeature([21 + 299], tf.float32),
										   'y' : tf.FixedLenFeature([5], tf.float32),
									   })

	x = tf.cast(features['x'], tf.float64)
	y = tf.cast(features['y'], tf.float64)
	#x = features['x']
	#y = features['y']

	return x, y



if __name__ == '__main__':
	train_x_path = "data/train_x0.csv"
	train_json_path = "data/transfer.json"
	train_y_path = "data/train_y.csv"

	create_record(train_x_path, train_y_path, train_json_path)

