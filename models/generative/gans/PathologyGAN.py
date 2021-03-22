from models.generative.gans.GAN import GAN
from models.networks.discriminator import *
from models.networks.generator import *
from models.evaluation.features import *
from data_manipulation.utils import *
from models.normalization import *
from models.regularizers import *
from models.activations import *
from models.evaluation import *
from models.optimizer import *
from models.utils import *
from models.tools import *
from models.loss import *
from models.ops import *

import tensorflow.compat.v1 as tf
import numpy as np


'''
GAN model combining features from BigGAN, StyleGAN, and Relativistic average iscriminator.
	1. Attention network: SAGAN/BigGAN.
	2. Orthogonal initalization and regularization: SAGAN/BigGAN.
	3. Spectral normalization: SNGAN/SAGAN/BigGAN.
	4. Mapping network: StyleGAN.
	5. Relativistic average discriminator.
'''
class PathologyGAN(GAN):
	def __init__(self,
				data,                       			# Dataset class, training and test data.
				z_dim,	                    			# Latent space dimensions.
				use_bn,                      			# Batch Normalization flag to control usage in discriminator.
				alpha,                       			# Alpha value for LeakyReLU.
				beta_1,                      			# Beta 1 value for Adam Optimizer.
				learning_rate_g,             			# Learning rate generator.
				learning_rate_d,             			# Learning rate discriminator.
				spectral=True,							# Spectral Normalization for weights.
				noise_input_f=True,						# Input noise at each generator layer.
				style_mixing=.5,						# Mixing probability threshold.
				layers=5,					 			# Number for layers for Generator/Discriminator.
				synth_layers=4, 			 			# Number for layers for Generator/Discriminator.
				attention=28,                			# Attention Layer dimensions, default after hegiht and width equal 28 to pixels.
				power_iterations=1,          			# Iterations of the power iterative method: Calculation of Eigenvalues, Singular Values.
				beta_2=None,                 			# Beta 2 value for Adam Optimizer.
				n_critic=1,                  			# Number of batch gradient iterations in Discriminator per Generator.
				gp_coeff=.5,                 			# Gradient Penalty coefficient for the Wasserstein Gradient Penalty loss.
				init = 'orthogonal',		 			# Weight Initialization: Orthogonal in BigGAN.
				loss_type='relativistic standard',     	# Loss function type: Standard, Least Square, Wasserstein, Wasserstein Gradient Penalty.
				regularizer_scale=1e-4,      			# Orthogonal regularization.
				model_name='PathologyGAN'   			# Name to give to the model.
				):

		# Architecture parameters.
		self.attention = attention
		self.layers = layers
		self.synth_layers = synth_layers
		self.normalization = conditional_instance_norm
		self.spectral = spectral
		self.noise_input_f = noise_input_f

		# Training parameters
		self.style_mixing = style_mixing
		self.power_iterations = power_iterations
		self.gp_coeff = gp_coeff
		self.beta_2 = beta_2
		self.regularizer_scale = regularizer_scale
		self.top_k_samples = int(data.batch_size/2.)
		
		super().__init__(data=data, z_dim=z_dim, use_bn=use_bn, alpha=alpha, beta_1=beta_1, learning_rate_g=learning_rate_g, learning_rate_d=learning_rate_d, 
						 conditional=False, n_critic=n_critic, init=init, loss_type=loss_type, model_name=model_name)

	# Model inputs
	def model_inputs(self):
		
		# Image input.
		real_images = tf.placeholder(dtype=tf.float32, shape=(None, self.image_width, self.image_height, self.image_channels), name='real_images')
		
		# Latent vectors
		z_input_1 = tf.placeholder(dtype=tf.float32, shape=(None, self.z_dim), name='z_input_1')
		z_input_2 = tf.placeholder(dtype=tf.float32, shape=(None, self.z_dim), name='z_input_2')
		
		# W Latent vectors
		w_latent_in = tf.placeholder(dtype=tf.float32, shape=(None, self.z_dim, self.layers+1), name='w_latent_in')

		# Learning rates
		learning_rate_g = tf.placeholder(dtype=tf.float32, name='learning_rate_g')
		learning_rate_d = tf.placeholder(dtype=tf.float32, name='learning_rate_d')

		# Probability rate of using style mixing regularization.
		style_mixing_prob = tf.placeholder(dtype=tf.float32, name='style_mixing_prob')

		return real_images, z_input_1, z_input_2, w_latent_in, learning_rate_g, learning_rate_d, style_mixing_prob

	# Discriminator Network.
	def discriminator(self, images, reuse, init, label_input=None, name='discriminator'):
		output, logits, feature_space = discriminator_resnet(images=images, layers=self.layers, spectral=self.spectral, activation=leakyReLU, reuse=reuse, attention=self.attention, init=init, feature_space_flag=True,
											  normalization=None,regularizer=orthogonal_reg(self.regularizer_scale), label=label_input, name=name)
		return output, logits, feature_space

	def mapping(self, z_input, reuse, is_train, normalization, init):
		w_latent = mapping_resnet(z_input=z_input, z_dim=self.z_dim, layers=self.synth_layers, reuse=reuse, is_train=is_train, spectral=self.spectral, activation=ReLU, normalization=normalization, 
								  init=init, regularizer=orthogonal_reg(self.regularizer_scale))
		return w_latent

	# Generator nerwork.
	def generator(self, w_latent, reuse, is_train, init):
		output = generator_resnet_style(w_input=w_latent, image_channels=self.image_channels, layers=self.layers, spectral=self.spectral, activation=leakyReLU, reuse=reuse, is_train=is_train,
										normalization=self.normalization, init=init, noise_input_f=self.noise_input_f, regularizer=orthogonal_reg(self.regularizer_scale), attention=self.attention)
		return output

	# Loss function.
	def loss(self):
		loss_dis, loss_gen = losses(self.loss_type, self.output_fake, self.output_real, self.logits_fake, self.logits_real, real_images=self.real_images, fake_images=self.fake_images, 
									top_k_samples=self.top_k_samples, discriminator=self.discriminator, gp_coeff=self.gp_coeff, init=self.init)

		return loss_dis, loss_gen

	# Optimizer.
	def optimization(self, gpus=[0]):
		train_discriminator, train_generator = optimizer(self.beta_1, self.loss_gen, self.loss_dis, self.loss_type, self.learning_rate_input_g, self.learning_rate_input_d, beta_2=self.beta_2, gpus=gpus)
		return train_discriminator, train_generator

	# Style mixing regularization.
	def style_mixing_reg(self, w_input_1, w_input_2, style_mixing_prob, layers):
		w_latent_1 = tf.tile(w_input_1[:,:, np.newaxis], [1, 1, layers+1])
		w_latent_2 = tf.tile(w_input_2[:,:, np.newaxis], [1, 1, layers+1])    
		with tf.variable_scope('style_mixing_reg'):			
			layers_index = 1 + layers
			possible_layers = np.arange(layers_index)[np.newaxis, np.newaxis, :]
			layer_cut = tf.cond(tf.random_uniform([], 0.0, 1.0) < style_mixing_prob, lambda: tf.random.uniform([], 1, layers_index, dtype=tf.int32), 
																					 lambda: tf.constant(layers_index, dtype=tf.int32))
		w_latent = tf.where(tf.broadcast_to(possible_layers<layer_cut, tf.shape(w_latent_1)), w_latent_1, w_latent_2)
		return w_latent

	def style_mixing_reg2(self, w_input_1, w_input_2, style_mixing_prob, layers):
		w_latent_1 = tf.tile(w_input_1[:,:, np.newaxis], [1, 1, layers+1])
		w_latent_2 = tf.tile(w_input_2[:,:, np.newaxis], [1, 1, layers+1])    
		with tf.variable_scope('style_mixing_reg'):			
			layers_index = 1 + layers
			stages = np.linspace(0.0, 1.0, layers_index)
			w_interpolation = w_latent_1*stages +w_latent_2*(1-stages) 
			w_latent = tf.cond(tf.random_uniform([], 0.0, 1.0)<style_mixing_prob, lambda:w_latent_1, lambda:w_interpolation)
		return w_latent

	# Put together the whole GAN.
	def build_model(self):

		with tf.device('/gpu:0'):
			# Inputs.
			self.real_images, self.z_input_1, self.z_input_2, self.w_latent_in, self.learning_rate_input_g, self.learning_rate_input_d, self.style_mixing_prob = self.model_inputs()

			# Neural Network Generator and Discriminator.
			self.w_latent_1 = self.mapping(self.z_input_1, reuse=False, is_train=True, normalization=None, init=self.init)
			self.w_latent_2 = self.mapping(self.z_input_2, reuse=True, is_train=True, normalization=None, init=self.init)
			self.w_latent = self.style_mixing_reg2(self.w_latent_1, self.w_latent_2, self.style_mixing_prob, self.layers)
			self.fake_images = self.generator(self.w_latent, reuse=False, is_train=True, init=self.init)

		i=0
		if self.image_height > 256: i = 0 
		with tf.device('/gpu:%s' % i):

			self.w_latent_out = self.mapping(self.z_input_1, reuse=True, is_train=False, normalization=None, init=self.init)
			self.output_gen = self.generator(self.w_latent_in, reuse=True, is_train=False, init=self.init)

		if self.image_height > 256: i += 1
		with tf.device('/gpu:%s' % i):
			# Discriminator.
			self.output_fake, self.logits_fake, self.feature_space_fake = self.discriminator(images=self.fake_images, reuse=False, init=self.init) 

		if self.image_height > 256: i = 1
		with tf.device('/gpu:%s' % i):
			self.output_real, self.logits_real, self.feature_space_real = self.discriminator(images=self.real_images, reuse=True, init=self.init)

			# Losses.
			self.loss_dis, self.loss_gen = self.loss()

		if self.image_height > 256: 
			gpus = [2,3]
		else:
			gpus = [0]
		# Optimizers.
		self.train_discriminator, self.train_generator = self.optimization(gpus=gpus)

	# Train function.
	def train(self, epochs, data_out_path, data, restore, print_epochs=10, n_images=25, checkpoint_every=None, report=False):
		run_epochs = 0    
		saver = tf.train.Saver()

		# Setups.
		checkpoints, csvs = setup_output(data_out_path=data_out_path, model_name=self.model_name, restore=restore)
		losses = ['Generator Loss', 'Discriminator Loss']
		setup_csvs(csvs=csvs, model=self, losses=losses)
		report_parameters(self, epochs, restore, data_out_path)

		# Training session.
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
		
		with tf.Session(config=config) as session:
			session.run(tf.global_variables_initializer())

			# Restore previous session.
			if restore:
				check = get_checkpoint(data_out_path)
				saver.restore(session, check)
				print('Restored model: %s' % check)

			# Saving graph details.
			# writer = tf.summary.FileWriter(os.path.join(data_out_path, 'tensorboard'), graph_def=session.graph_def)	

			# Steady latent input
			steady_latent_1 = np.random.normal(size=(n_images, self.z_dim)) 
			feed_dict = {self.z_input_1: steady_latent_1}
			w_latent_out = session.run([self.w_latent_out], feed_dict=feed_dict)[0]
			w_latent_in = np.tile(w_latent_out[:,:, np.newaxis], [1, 1, self.layers+1])

			# Epoch Iteration.
			for epoch in range(1, epochs+1):

				saver.save(sess=session, save_path=checkpoints)
				
				# Batch Iteration.
				for batch_images, batch_labels in data.training:
					# Inputs.
					z_batch_1 = np.random.normal(size=(self.batch_size, self.z_dim)) 
					z_batch_2 = np.random.normal(size=(self.batch_size, self.z_dim)) 

					# Feed inputs.
					feed_dict = {self.z_input_1:z_batch_1, self.z_input_2:z_batch_2, self.w_latent_in:w_latent_in, self.real_images:batch_images, self.style_mixing_prob:self.style_mixing,
								self.learning_rate_input_g: self.learning_rate_g, self.learning_rate_input_d: self.learning_rate_d}
					
					# Update critic.
					session.run([self.train_discriminator], feed_dict=feed_dict, options=run_options)
					
					# Update generator after n_critic updates from discriminator.
					if run_epochs%self.n_critic == 0:
						session.run([self.train_generator], feed_dict=feed_dict, options=run_options)

		            # Print losses and Generate samples.
					if run_epochs % print_epochs == 0:
						epoch_loss_dis, epoch_loss_gen = session.run([self.loss_dis, self.loss_gen], feed_dict=feed_dict)
						update_csv(model=self, file=csvs[0], variables=[epoch_loss_gen, epoch_loss_dis], epoch=epoch, iteration=run_epochs, losses=losses)

					run_epochs += 1
					# break
					
				# Save model.
				saver.save(sess=session, save_path=checkpoints)
				data.training.reset()

				############################### FID TRACKING ##################################################
				# Save checkpoint and generate images for FID every X epochs.
				if (checkpoint_every is not None and epoch % checkpoint_every == 0) or (epochs==epoch):
					generate_samples_epoch(session=session, model=self, data=data, epoch=epoch, data_out_path=data_out_path, report=report)

				########################### IMAGE GENERATION EPOCH ############################################
				# Random sample images.
				feed_dict = {self.z_input_1: np.random.normal(size=(64, self.z_dim))}
				w_latent_out = session.run([self.w_latent_out], feed_dict=feed_dict)[0]
				w_latent_in = np.tile(w_latent_out[:,:, np.newaxis], [1, 1, self.layers+1])
				feed_dict = {self.w_latent_in:w_latent_in, self.real_images:batch_images}
				gen_samples = session.run([self.output_gen], feed_dict=feed_dict)[0]
				write_sprite_image(filename=os.path.join(data_out_path, 'images/gen_samples_epoch_%s.png' % epoch), data=gen_samples, metadata=False)

				# Steady latent vector and image generation.
				feed_dict = {self.z_input_1: steady_latent_1}
				w_latent_out = session.run([self.w_latent_out], feed_dict=feed_dict)[0]
				w_latent_in = np.tile(w_latent_out[:,:, np.newaxis], [1, 1, self.layers+1])
				feed_dict = {self.w_latent_in:w_latent_in, self.real_images:batch_images}
				gen_samples = session.run([self.output_gen], feed_dict=feed_dict)[0]
				write_sprite_image(filename=os.path.join(data_out_path, 'images/gen_samples_steady_epoch_%s.png' % epoch), data=gen_samples, metadata=False)