
from sys import argv
from model import *
from read_data import *
from tensorflow.python.framework import graph_util
from tfrc import read_and_decode 

batch_size = 128
dropout = 0.4



train_path = "data/train.tfrecords"
test_path = "data/test.tfrecords"

def normalize(x):
	u = np.mean(x,0)
	sig = np.std(x,0)

	new_x = (x - u) / sig

	return new_x

def get_train_test_data(train_path, test_path, batch_size):
		ecgdata, label = read_and_decode(train_path)
		test_data_batch, test_label_batch = read_and_decode(test_path)


		train_data, train_label = tf.train.shuffle_batch([ecgdata, label],
													batch_size=batch_size, capacity=50000,
													#num_threads=3,
													min_after_dequeue=1000)

		test_data, test_label = tf.train.shuffle_batch([test_data_batch, test_label_batch],
													batch_size=128, capacity=50000,
													#num_threads=3,
													min_after_dequeue=1000)

		return train_data, train_label, test_data, test_label 

train_data, train_label, test_data, test_label = get_train_test_data(train_path, test_path, batch_size)

config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
version = tf.constant('v1.0.0', name='version')

#with tf.device('/gpu:1'):
model = build_network()


saver=tf.train.Saver()
config.gpu_options.per_process_gpu_memory_fraction = 0.4


read_log = argv[1]
prefix = 'mnbp_v1'
epochs = 25000
predict = True

if predict:
		tx = np.loadtxt("data/test_x0.csv", delimiter=',', usecols=range(1,22))
		ph = np.array([[125.3645,77.016500,1.606788,1.372469,2.768998] for x in range(256)])

init_global = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()
with tf.Session(config=config) as sess:
	sess.run(init_global)
	#sess.run(init_local)

	#tf.summary.FileWriter('log', sess.graph)

	if read_log == "log":			
		with open("log/checkpoint",'r') as f1:
				txt = f1.readline()
				point = txt.strip().replace('model_checkpoint_path: ','').replace("\"",'')
				print point
				saver.restore(sess,"log/%s"%point)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	print 'training start...'
	for step in range(epochs):
		batch_x, batch_y = sess.run([train_data, train_label])
		#batch_x = normalize(batch_x)

		sess.run(model.optimizer, feed_dict={model.x: batch_x, model.labels: batch_y, model.dropout: dropout})


		if step % 10 == 0:	
			train_logits, loss = sess.run((model.logits, model.loss), feed_dict={model.x: batch_x, model.labels: batch_y, model.dropout: dropout})

			test_x, test_y = sess.run([test_data, test_label])
			#test_x = normalize(test_x)
			mid, test_logits, val_loss = sess.run([model.mid, model.logits, model.loss], feed_dict={model.x: test_x, model.labels: test_y, model.dropout: 0})
			
			#rand_acc = np.mean(np.square(np.log1p(ph) - np.log1p(test_y)))
			acc = np.mean(np.square(np.log1p(train_logits) - np.log1p(batch_y)))
			val_acc = np.mean(np.square(np.log1p(test_logits) - np.log1p(test_y)))
			
			print "Epoch %d/%d - loss: %f \tval_loss: %f\tacc: %f\tval_acc: %f"  % (step+1,epochs, loss, val_loss, acc, val_acc) 
			#print "Epoch %d/%d - loss: %f \tval_loss: %f\tacc: %f\tval_acc: %f\trand_acc: %f"  % (step+1,epochs, loss, val_loss, acc, val_acc, rand_acc) 

			

		if step % 200 == 0:	
			print test_logits[10:20]
			#print mid
			print '\n'
			print np.mean(test_logits,0)
		if step % 1000 == 199:	
			checkpoint_filepath='log/step-%d.ckpt' % step
			saver.save(sess,checkpoint_filepath)
			print '\n~~~~checkpoint saved!~~~~~\n'

		if step % 1000 == 1 and step > epochs - 2000 and False:	
			output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, 
											output_node_names=["x", "y", 'dropout', 'version'])
			with tf.gfile.FastGFile('./load_pb/%s_%d.pb' %(prefix,step), mode='wb') as f:
				f.write(output_graph_def.SerializeToString())

		if step % 10 == 0 and val_acc<0.02:
			tx = normalize(tx)
			for i in range(0,9600,200):
				px = tx[i:i+200]
				res = sess.run(model.logits, feed_dict={model.x: px, model.dropout: 0})
			
				if i == 0:
					pred_out = res
				else:
					pred_out = np.concatenate((pred_out, res))	

			print pred_out.shape
			np.savetxt("data/submit_v2_%s.csv" % str(round(val_acc,4)), np.round(pred_out,3),  delimiter=',', fmt='%f')
			

	coord.request_stop()
	coord.join(threads)



#json.dump(val_acc_log, open("training_log.json",'w'))




