
import os
import time
import tensorflow as tf
from ops import *
from utils import *
import numpy as np

class DCGAN(object):
	def __init__(self, sess, image_size=96, is_crop=False,
				 batch_size=64, sample_size=64,
				 z_dim=100, gf_dim=64, df_dim=64,
				 gfc_dim=1024, dfc_dim=1024, c_dim=1,
				 checkpoint_dir=None):
		"""

		Args:
			sess: TensorFlow session
			batch_size: The size of batch. Should be specified before training.
			z_dim: (optional) Dimension of dim for Z. [100]
			gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
			df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
			gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
			dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
			c_dim: (optional) Dimension of image color. [3]
		"""
		# TODO: Implementation for 96x96 images	
	
		self.sess = sess
		self.is_crop = is_crop
		self.batch_size = batch_size
		self.image_size = image_size
		self.sample_size = sample_size
		self.image_shape = [image_size, image_size, c_dim]

		self.z_dim = z_dim

		self.gf_dim = gf_dim
		self.df_dim = df_dim

		self.gfc_dim = gfc_dim
		self.dfc_dim = dfc_dim

		self.c_dim = c_dim

		# batch normalization : deals with poor initialization helps gradient flow
		self.d_bn1 = batch_norm(name='d_bn1')
		self.d_bn2 = batch_norm(name='d_bn2')
		self.d_bn3 = batch_norm(name='d_bn3')
		self.d_bn4 = batch_norm(name='d_bn4')

		self.g_bn0 = batch_norm(name='g_bn0')
		self.g_bn1 = batch_norm(name='g_bn1')
		self.g_bn2 = batch_norm(name='g_bn2')
		self.g_bn3 = batch_norm(name='g_bn3')

		self.checkpoint_dir = checkpoint_dir
		self.build_model()

		self.model_name = "DCGAN.model"

	def build_model(self):

		# placeholders for model input
		self.images = tf.placeholder(tf.float32, [None] + self.image_shape, name='real_images')
		self.sample_images= tf.placeholder(tf.float32, [None] + self.image_shape, name='sample_images')
		self.labels = tf.placeholder(tf.float32,[None,5], name = 'labels')
		
		self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
		self.z_sum = tf.histogram_summary("z", self.z)

		# generate image
		self.G = self.generator(self.z)
		# pass real image through discriminator

		# Modification of loss function required for classification
		self.D_logits, self.features = self.discriminator(self.images)

		self.sampler = self.sampler(self.z)
		
		#pass fake image through discriminator
		self.D_logits_ ,self.features_ = self.discriminator(self.G, reuse=True)

		#self.d_sum = tf.histogram_summary("d", self.D)
		#self.d__sum = tf.histogram_summary("d_", self.D_)
		self.G_sum = tf.image_summary("G", self.G)

		# losses
		self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits,self.labels))
		self.d_loss_fake = tf.reduce_mean(tf.nn.softplus(log_sum_exp(self.D_logits_)))

		self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
		self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)

		# generator loss

		self.g_loss = tf.reduce_mean(tf.abs(tf.reduce_mean(self.features,0) - tf.reduce_mean(self.features_,0)))

		# discriminator loss
		self.weight_fake = 1.0;
		self.d_loss = self.d_loss_real + self.weight_fake*self.d_loss_fake

		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.D_logits,1),tf.argmax(self.labels,1)),tf.float32))


		self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
		self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

		t_vars = tf.trainable_variables()

		self.d_vars = [var for var in t_vars if 'd_' in var.name]
		self.g_vars = [var for var in t_vars if 'g_' in var.name]

		self.saver = tf.train.Saver(max_to_keep=1)

	def train(self, config):
		I,T= self.get_training_data()
		assert I.shape[1:] == (96,96,1)
		
		train_size = int(I.shape[0]*0.8)
		imgs = I[:train_size,:,:,:]
		targ = T[:train_size,:]

		val_x = I[train_size:,:,:,:]
		val_y = T[train_size:,:]



		print 'Training data loaded'

		d_optim = tf.train.AdamOptimizer(config.learning_rate*10, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
		#d_optim = tf.train.GradientDescentOptimizer(0.001).minimize(self.d_loss, var_list=self.d_vars)
		g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
		
		tf.initialize_all_variables().run()

		self.g_sum = tf.merge_summary([self.z_sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
		self.d_sum = tf.merge_summary([self.z_sum, self.d_loss_real_sum, self.d_loss_sum])
		self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)

		sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))

		counter = 1
		start_time = time.time()

		print 'Starting iterations...'
		for epoch in xrange(config.epoch):
			perm = np.random.permutation(imgs.shape[0])
			imgs = imgs[perm,:,:,:]
			targ = targ[perm,:]
			num_batches = int(imgs.shape[0]/self.batch_size)

			for idx in xrange(0, num_batches):
				
				idxs = idx*self.batch_size
				idxe = idxs + self.batch_size
				batch_images = imgs[idxs:idxe,:,:,:] 
				batch_y = targ[idxs:idxe,:]

				batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

				# Update D network
				_, summary_str = self.sess.run([d_optim, self.d_sum],feed_dict={ self.images: batch_images, self.labels: batch_y, self.z: batch_z })
				self.writer.add_summary(summary_str, counter)

				# Update G network
				_, summary_str = self.sess.run([g_optim, self.g_sum],feed_dict={ self.images: batch_images, self.z: batch_z })
				self.writer.add_summary(summary_str, counter)

				# Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
				_, summary_str = self.sess.run([g_optim, self.g_sum],feed_dict={ self.images: batch_images, self.z: batch_z })
				self.writer.add_summary(summary_str, counter)

				errD_fake = self.d_loss_fake.eval({self.images: batch_images, self.z: batch_z})
				errD_real = self.d_loss_real.eval({self.images: batch_images, self.labels: batch_y})
				errG = self.g_loss.eval({self.images: batch_images, self.z: batch_z})
				acc = self.accuracy.eval({self.images: batch_images, self.labels: batch_y})
				counter += 1

				print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, accuracy: %.4f" % (epoch+1, idx+1, num_batches,time.time() - start_time, errD_fake+errD_real, errG, acc))

				if (idx+1) % 100 == 0:
					samples, d_loss, g_loss = self.sess.run([self.sampler, self.d_loss, self.g_loss],feed_dict={self.z: sample_z, self.images: imgs[:self.sample_size,:,:,:], self.labels: targ[:self.sample_size,:]})
					save_images(samples, [8, 8],'./samples/train_{:02d}_{:04d}.png'.format(epoch, idx))
					print("Epoch: %02d, Iter: %03d, d_loss: %.8f, Val accuracy: %.4f" % (epoch+1, idx+1, d_loss, self.get_accuracy(val_x, val_y)))

				if np.mod(counter, 500) == 2:
					self.save(config.checkpoint_dir, counter)

		print 'Training Done\nLoading test data...'
		img_t,label_t = self.get_test_data()
		print 'Test data loaded\nCalculating accuracy...'
		print 'Test accuracy: %.4f' % (self.get_accuracy(img_t,label_t))



	def get_accuracy(self, X, Y):
	    batch_size = 64
	    num_batches = int(X.shape[0]/batch_size)
	    acc = 0
	    for i in xrange(num_batches):
	        idxs = i*batch_size
	        idxe = idxs + batch_size
	        batch_x = X[idxs:idxe,:,:,:] 
	        batch_y = Y[idxs:idxe,:]
	        acc += self.sess.run(self.accuracy, feed_dict={self.images: batch_x, self.labels: batch_y})

	    acc /= num_batches
	    return acc

	def discriminator(self, image, reuse=False):

		# df_dim = 64
		# 5 conv layers, 1 fully connected layer 
		if reuse:
			tf.get_variable_scope().reuse_variables()

		h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
		h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
		h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
		h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
		h4 = lrelu(self.d_bn4(conv2d(h3, self.df_dim*8, name='d_h4_conv')))
		shape = int(np.prod(h4.get_shape()[1:]))
		#print h3.get_shape()
		h5 = linear(tf.reshape(h4, [-1, shape]), 5, 'd_h3_lin')   # 5 classes

		return h5,h4

	def generator(self, z):

		# 1 fully connected layer, 4 conv_transpose layers

		self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*4*4, 'g_h0_lin', with_w=True)

		self.h0 = tf.reshape(self.z_, [-1, 4, 4, self.gf_dim * 8])
		h0 = tf.nn.relu(self.g_bn0(self.h0))

		self.h1, self.h1_w, self.h1_b = conv2d_transpose(h0,[self.batch_size, 8, 8, self.gf_dim*4], name='g_h1', with_w=True)
		h1 = tf.nn.relu(self.g_bn1(self.h1))

		h2, self.h2_w, self.h2_b = conv2d_transpose(h1,[self.batch_size, 16, 16, self.gf_dim*2], name='g_h2', with_w=True)
		h2 = tf.nn.relu(self.g_bn2(h2))

		h3, self.h3_w, self.h3_b = conv2d_transpose(h2,[self.batch_size, 32, 32, self.gf_dim*1], name='g_h3', with_w=True)
		h3 = tf.nn.relu(self.g_bn3(h3))

		h4, self.h4_w, self.h4_b = conv2d_transpose(h3,[self.batch_size, 96, 96, self.c_dim], d_h = 3, d_w = 3, name='g_h4', with_w=True)

		return tf.nn.tanh(h4)

	def sampler(self, z, y=None):
		tf.get_variable_scope().reuse_variables()

		h0 = tf.reshape(linear(z, self.gf_dim*8*4*4, 'g_h0_lin'),[-1, 4, 4, self.gf_dim * 8])
		h0 = tf.nn.relu(self.g_bn0(h0, train=False))

		h1 = conv2d_transpose(h0, [self.batch_size, 8, 8, self.gf_dim*4], name='g_h1')
		h1 = tf.nn.relu(self.g_bn1(h1, train=False))

		h2 = conv2d_transpose(h1, [self.batch_size, 16, 16, self.gf_dim*2], name='g_h2')
		h2 = tf.nn.relu(self.g_bn2(h2, train=False))

		h3 = conv2d_transpose(h2, [self.batch_size, 32, 32, self.gf_dim*1], name='g_h3')
		h3 = tf.nn.relu(self.g_bn3(h3, train=False))

		h4 = conv2d_transpose(h3, [self.batch_size, 96, 96, self.c_dim], d_h = 3, d_w = 3, name='g_h4')

		return tf.nn.tanh(h4)

	def save(self, checkpoint_dir, step):
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name), global_step=step)

	def load(self, checkpoint_dir):
		print 'Reading checkpoints...'

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
			return True
		else:
			return False

	def get_training_data(self):
		fid_images = open('../data/NORB/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat','r')
		fid_labels = open('../data/NORB/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat','r')

		for i in xrange(6):
			a = fid_images.read(4)    #header

		num_images = 24300*2
		images = np.zeros((num_images,96,96))

		for idx in xrange(num_images):
			temp = fid_images.read(96*96)
			images[idx,:,:] = np.fromstring(temp,'uint8').reshape(96,96).T 

		for i in xrange(5):
			a = fid_labels.read(4) #header

		labels = np.fromstring(fid_labels.read(num_images*np.dtype('int32').itemsize),'int32')
		labels = np.repeat(labels,2)

		perm = np.random.permutation(num_images)
		images = images[perm]
		labels = labels[perm]
		labels = labels.reshape(images.shape[0],1) == np.arange(5) # one hot
		
		return images[:,:,:,np.newaxis]/255.0,labels 

	def get_test_data(self):
		fid_images = open('../data/NORB/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat','r')
		fid_labels = open('../data/NORB/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat','r')

		for i in xrange(6):
			a = fid_images.read(4)    #header

		num_images = 24300*2
		images = np.zeros((num_images,96,96))

		for idx in xrange(num_images):
			temp = fid_images.read(96*96)
			images[idx,:,:] = np.fromstring(temp,'uint8').reshape(96,96).T 

		for i in xrange(5):
			a = fid_labels.read(4) #header

		labels = np.fromstring(fid_labels.read(num_images*np.dtype('int32').itemsize),'int32')
		labels = np.repeat(labels,2)

		perm = np.random.permutation(num_images)
		images = images[perm]
		labels = labels[perm]
		labels = labels.reshape(images.shape[0],1) == np.arange(5) # one hot
		#imshow(images[2331,:,:])
		
		return images[:,:,:,np.newaxis]/255.0,labels 

