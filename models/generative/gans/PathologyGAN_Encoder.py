from models.generative.gans.GAN import GAN
from models.networks.discriminator import *
from models.networks.generator import *
from models.networks.encoder_gan import *
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


class PathologyGAN_Encoder(GAN):
	def __init__(self,
				data,                       			# Dataset type, training, validatio, and test data.
				z_dim,	                    			# Latent space dimensions.
				use_bn,                      			# Batch Normalization flag to control usage in discriminator.
				alpha,                       			# Alpha value for LeakyReLU.
				beta_1,                      			# Beta 1 value for Adam Optimizer.
				learning_rate_g,             			# Learning rate generator.
				learning_rate_d,             			# Learning rate discriminator.
				learning_rate_e=None,             		# Learning rate encoder.
				spectral=True,							# Spectral Normalization for weights.
				noise_input_f=True,						# Input noise at each generator layer.
				style_mixing=.5,						# Mixing probability threshold.
				layers=5,					 			# Number for layers for Generator/Discriminator.
				synth_layers=4, 			 			# Number for layers for Generator/Discriminator.
				attention=28,                			# Attention Layer dimensions, default after hegiht and width equal 28 to pixels.
				power_iterations=1,          			# Iterations of the power iterative method: Calculation of Eigenvalues, Singular Values.
				beta_2=None,                 			# Beta 2 value for Adam Optimizer.
				n_critic=1,                  			# Number of batch gradient iterations in Discriminator per Generator.
				gp_coeff=.5,                 			# Gradient Penalty coefficient for Discriminator, avoid big updates.
				init = 'orthogonal',		 			# Weight Initialization: default Orthogonal.
				loss_type='relativistic standard',     	# Loss function type: default Relativistic Average Discriminator.
				regularizer_scale=1e-4,      			# Orthogonal regularization.
				model_name='PathologyGAN_Encoder'   	# Model Name.
				):

		# Architecture parameters.
		self.attention = attention
		self.layers = layers
		self.synth_layers = synth_layers
		self.normalization = conditional_instance_norm
		self.spectral = spectral
		self.noise_input_f = noise_input_f

		# Hyperparameters.
		self.style_mixing = style_mixing
		self.power_iterations = power_iterations
		self.gp_coeff = gp_coeff
		self.beta_2 = beta_2
		self.regularizer_scale = regularizer_scale
		if learning_rate_e is None:
			self.learning_rate_e = learning_rate_d
		else:
			self.learning_rate_e = learning_rate_e
		self.top_k_samples = int(data.batch_size/2.)
		
		# Super Initializer.
		super().__init__(data=data, z_dim=z_dim, use_bn=use_bn, alpha=alpha, beta_1=beta_1, learning_rate_g=learning_rate_g, learning_rate_d=learning_rate_d, 
						 conditional=False, n_critic=n_critic, init=init, loss_type=loss_type, model_name=model_name)

	# GAN inputs
	def model_inputs(self):
		
		# Image input.
		real_images_1 = tf.placeholder(dtype=tf.float32, shape=(None, self.image_width, self.image_height, self.image_channels), name='real_images_1')
		real_images_2 = tf.placeholder(dtype=tf.float32, shape=(None, self.image_width, self.image_height, self.image_channels), name='real_images_2')
		# Latent vectors.
		z_input_1 = tf.placeholder(dtype=tf.float32, shape=(None, self.z_dim), name='z_input_1')
		z_input_2 = tf.placeholder(dtype=tf.float32, shape=(None, self.z_dim), name='z_input_2')
		# W Latent vectors.
		w_latent_in = tf.placeholder(dtype=tf.float32, shape=(None, self.z_dim, self.layers+1), name='w_latent_in')
		w_latent_ref = tf.placeholder(dtype=tf.float32, shape=(None, self.z_dim, self.layers+1), name='w_latent_ref')
		# Learning rates.
		learning_rate_g = tf.placeholder(dtype=tf.float32, name='learning_rate_g')
		learning_rate_d = tf.placeholder(dtype=tf.float32, name='learning_rate_d')
		learning_rate_e = tf.placeholder(dtype=tf.float32, name='learning_rate_e')
		# Probability rate of using style mixing regularization.
		style_mixing_prob = tf.placeholder(dtype=tf.float32, name='style_mixing_prob')

		return real_images_1, real_images_2, z_input_1, z_input_2, w_latent_in, w_latent_ref, learning_rate_g, learning_rate_d, learning_rate_e, style_mixing_prob

	# Encoder Network.
	def encoder(self, images, reuse, is_train, init):
		# w_latent = encoder_resnet_incr(images=images, z_dim=self.z_dim, layers=self.layers, spectral=self.spectral, activation=leakyReLU, reuse=reuse, init=init, 
									   # is_train=is_train, regularizer=orthogonal_reg(self.regularizer_scale), normalization=None, attention=self.attention)
		w_latent = encoder_resnet_instnorm(images=images, latent_dim=self.z_dim, layers=self.layers, spectral=self.spectral, 
									   activation=leakyReLU, reuse=reuse, is_train=is_train, init=init, regularizer=orthogonal_reg(self.regularizer_scale), 
									   normalization=instance_norm, attention=self.attention, name='encoder')						
		return w_latent

	# Mapping Network.
	def mapping(self, z_input, reuse, is_train, normalization, init):
		w_latent = mapping_resnet(z_input=z_input, z_dim=self.z_dim, layers=self.synth_layers, reuse=reuse, is_train=is_train, spectral=self.spectral, activation=ReLU, 
								  normalization=normalization, init=init, regularizer=orthogonal_reg(self.regularizer_scale))
		return w_latent

	# Generator Network.
	def generator(self, w_latent, reuse, is_train, init):
		output = generator_resnet_style(w_input=w_latent, image_channels=self.image_channels, layers=self.layers, spectral=self.spectral, activation=leakyReLU, reuse=reuse, 
										is_train=is_train, normalization=self.normalization, init=init, noise_input_f=self.noise_input_f, attention=self.attention,
										regularizer=orthogonal_reg(self.regularizer_scale))
		return output

	# Discriminator Network.
	def discriminator(self, images, reuse, init, name, label_input=None):
		output, logits, feature_space = discriminator_resnet(images=images, layers=self.layers, spectral=self.spectral, activation=leakyReLU, reuse=reuse, attention=self.attention, 
															 normalization=None, feature_space_flag=True, init=init, regularizer=orthogonal_reg(self.regularizer_scale), label=label_input, 
															 name=name)
		return output, logits, feature_space

	# Loss Function.
	def loss(self):
		loss_dis, loss_gen = losses(self.loss_type, self.output_fake, self.output_real, self.logits_fake, self.logits_real, real_images=self.real_images_1, fake_images=self.fake_images, 
									top_k_samples=self.top_k_samples, discriminator=self.discriminator, gp_coeff=self.gp_coeff, init=self.init, dis_name='discriminator_gen')

		# MSE on Reference W latent and reconstruction, normalized by the dimensionality of the z vector.
		dimensionality_latent = self.z_dim
		latent_recon_error = tf.reduce_mean(tf.square(self.w_latent_ref-self.w_latent_e_1), axis=[-1])
		latent_recon_error = tf.reduce_sum(latent_recon_error, axis=[-1])
		loss_enc = tf.reduce_mean(latent_recon_error)/float(dimensionality_latent)

		return loss_dis, loss_gen, loss_enc

	# Optimizer.
	def optimization(self):
		train_discriminator, train_generator, _ = optimizer(self.beta_1, self.loss_gen, self.loss_dis, self.loss_type, self.learning_rate_input_d, 
															self.learning_rate_input_g, None, beta_2=self.beta_2, gen_name='generator', dis_name='discriminator_gen', 
															mapping_name='mapping_', encoder_name='encoder')
		
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			# Quick dirty optimizer for Encoder.
			trainable_variables = tf.trainable_variables()
			encoder_variables = [variable for variable in trainable_variables if variable.name.startswith('encoder')]
			train_encoder = tf.train.AdamOptimizer(learning_rate=self.learning_rate_input_e, beta1=self.beta_1).minimize(self.loss_enc, var_list=encoder_variables)

		return train_discriminator, train_generator, train_encoder

	# Style Mixing Regularization.
	def style_mixing_reg(self, w_input_1, w_input_2, style_mixing_prob, layers):
		w_latent_1 = tf.tile(w_input_1[:,:, np.newaxis], [1, 1, layers+1])
		w_latent_2 = tf.tile(w_input_2[:,:, np.newaxis], [1, 1, layers+1])    
		with tf.variable_scope('style_mixing_reg'):			
			layers_index = 1 + layers
			possible_layers = np.arange(layers_index)[np.newaxis, np.newaxis, :]
			layer_cut = tf.cond(tf.random_uniform([], 0.0, 1.0) < style_mixing_prob, lambda: tf.random.uniform([], 1, layers_index, dtype=tf.int32), lambda: tf.constant(layers_index, dtype=tf.int32))
		w_latent = tf.where(tf.broadcast_to(possible_layers<layer_cut, tf.shape(w_latent_1)), w_latent_1, w_latent_2)
		return w_latent

	# Build the GAN.
	def build_model(self):
		with tf.device('/gpu:0'):
			# Inputs.
			self.real_images_1, self.real_images_2, self.z_input_1, self.z_input_2, self.w_latent_in, self.w_latent_ref, self.learning_rate_input_g, self.learning_rate_input_d, self.learning_rate_input_e, self.style_mixing_prob = self.model_inputs()

			################### TRAINING #####################################################################################################################################
			# Generator Training.
			self.w_latent_1 = self.mapping(self.z_input_1, reuse=False, is_train=True, normalization=None, init=self.init)
			self.w_latent_2 = self.mapping(self.z_input_2, reuse=True, is_train=True, normalization=None, init=self.init)
			self.w_latent = self.style_mixing_reg(self.w_latent_1, self.w_latent_2, self.style_mixing_prob, self.layers)
			self.fake_images = self.generator(self.w_latent, reuse=False, is_train=True, init=self.init)
			# Discriminator Training.
			self.output_fake, self.logits_fake, self.feature_space_fake = self.discriminator(images=self.fake_images, reuse=False, init=self.init, name='discriminator_gen') 
			self.output_real, self.logits_real, self.feature_space_real = self.discriminator(images=self.real_images_1, reuse=True, init=self.init, name='discriminator_gen')
			# Encoder Training.		
			self.w_latent_e = self.encoder(images=self.real_images_2, reuse=False, is_train=True, init=self.init)
			self.w_latent_e_1 = tf.tile(self.w_latent_e[:,:, np.newaxis], [1, 1, self.layers+1])
			
			################### INFERENCE #####################################################################################################################################
			# Generator Inference.
			self.w_latent_out = self.mapping(self.z_input_1, reuse=True, is_train=False, normalization=None, init=self.init)
			self.output_gen = self.generator(self.w_latent_in, reuse=True, is_train=False, init=self.init)
			# Encoder Inference.
			self.w_latent_e_out = self.encoder(images=self.real_images_2, reuse=True, is_train=False, init=self.init)

			################### LOSS & OPTIMIZER ##############################################################################################################################
			# Losses.
			self.loss_dis, self.loss_gen, self.loss_enc = self.loss()
			# Optimizers.
			self.train_discriminator, self.train_generator, self.train_encoder  = self.optimization()

	# Training function. 
	def train(self, epochs, data_out_path, data, restore, print_epochs=10, n_images=25, checkpoint_every=None, report=False):
		run_epochs = 0    
		saver = tf.train.Saver()

		# Setups.
		checkpoints, csvs = setup_output(data_out_path=data_out_path, model_name=self.model_name, restore=restore)
		losses = ['Generator Loss', 'Discriminator Loss', 'Encoder Loss']
		setup_csvs(csvs=csvs, model=self, losses=losses)
		report_parameters(self, epochs, restore, data_out_path)

		# Session Options.
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

		# Training session.
		with tf.Session(config=config) as session:
			session.run(tf.global_variables_initializer())

			# Restore previous session.
			if restore:
				check = get_checkpoint(data_out_path)
				saver.restore(session, check)
				print('Restored model: %s' % check)

			# Saving graph details.
			writer = tf.summary.FileWriter(os.path.join(data_out_path, 'tensorboard'), graph_def=session.graph_def)	

			# Steady latent input
			batch_images = np.ones((self.batch_size, self.image_height, self.image_width, self.image_channels))
			steady_latent_1 = np.random.normal(size=(n_images, self.z_dim)) 
			feed_dict = {self.z_input_1: steady_latent_1, self.real_images_1:batch_images}
			w_latent_out = session.run([self.w_latent_out], feed_dict=feed_dict)[0]
			w_latent_in = np.tile(w_latent_out[:,:, np.newaxis], [1, 1, self.layers+1])

			# Epoch Iteration.
			for epoch in range(1, epochs+1):

				images_trained = 0
				
				# Batch Iteration.
				for batch_images, batch_labels in data.training:
					if images_trained >= 250000:
						break
					images_trained += batch_images.shape[0]

					# Inputs.
					z_batch_1 = np.random.normal(size=(self.batch_size, self.z_dim)) 
					z_batch_2 = np.random.normal(size=(self.batch_size, self.z_dim)) 

					################################# DISCRIMINATOR & GENERATOR ###############################################
					feed_dict = {self.z_input_1:z_batch_1, self.z_input_2:z_batch_2, self.w_latent_in:w_latent_in, self.real_images_1:batch_images, self.real_images_2:batch_images, 
								 self.style_mixing_prob:self.style_mixing, self.learning_rate_input_g: self.learning_rate_g, self.learning_rate_input_d: self.learning_rate_d, 
								 self.learning_rate_input_e: self.learning_rate_e}
					# Update discriminator.
					_, gen_img, w_latent_e_1  = session.run([self.train_discriminator, self.fake_images, self.w_latent], feed_dict=feed_dict, options=run_options)
					# Update generator.
					if run_epochs%self.n_critic == 0:
						session.run([self.train_generator], feed_dict=feed_dict)

					################################# ENCODER #################################################################
					# There's probably a better way, if all arrays are equal, then no there's no style mixing reg in this batch.
					flag_style_mixing = True
					for i in range(w_latent_e_1.shape[-1]-1):
						if not np.array_equal(w_latent_e_1[:,:,i], w_latent_e_1[:,:,i+1]):
							flag_style_mixing = False
							break

					if flag_style_mixing:
						feed_dict = {self.real_images_2:gen_img, self.w_latent_ref:w_latent_e_1, self.z_input_1:z_batch_1, self.z_input_2:z_batch_2, self.w_latent_in:w_latent_in, 
									 self.real_images_1:batch_images, self.learning_rate_input_e: self.learning_rate_e, self.style_mixing_prob:self.style_mixing, 
									 self.learning_rate_input_g: self.learning_rate_g, self.learning_rate_input_d: self.learning_rate_d}
						session.run([self.train_encoder], feed_dict=feed_dict, options=run_options)

					####################################################################################################
					# Print losses and Generate samples.
					if run_epochs % print_epochs == 0 and flag_style_mixing:
						model_outputs = [self.loss_gen, self.loss_dis, self.loss_enc]
						epoch_outputs = session.run(model_outputs, feed_dict=feed_dict, options=run_options)
						update_csv(model=self, file=csvs[0], variables=epoch_outputs, epoch=epoch, iteration=run_epochs, losses=losses)
					run_epochs += 1

				# Save model.
				saver.save(sess=session, save_path=checkpoints)
				data.training.reset()	
				
				############################### FID TRACKING ##################################################
				# Save checkpoint and generate images for FID every X epochs.
				if (checkpoint_every is not None and epoch % checkpoint_every == 0) or (epochs==epoch):
					generate_samples_epoch(session=session, model=self, data=data, epoch=epoch, data_out_path=data_out_path, report=report)

				########################### IMAGE GENERATION EPOCH ############################################
				# After each epoch dump a sample of generated images.
				z_batch_1 = np.random.normal(size=(64, self.z_dim)) 
				feed_dict = {self.z_input_1: z_batch_1}
				w_latent_out = session.run([self.w_latent_out], feed_dict=feed_dict)[0]
				w_latent_in = np.tile(w_latent_out[:,:, np.newaxis], [1, 1, self.layers+1])
				feed_dict = {self.w_latent_in:w_latent_in}
				gen_samples = session.run([self.output_gen], feed_dict=feed_dict)[0]
				write_sprite_image(filename=os.path.join(data_out_path, 'images/gen_samples_epoch_%s.png' % epoch), data=gen_samples, metadata=False)

				# Steady latent vector and image generation.
				feed_dict = {self.z_input_1: steady_latent_1, self.real_images_1:batch_images}
				w_latent_out = session.run([self.w_latent_out], feed_dict=feed_dict)[0]
				w_latent_in = np.tile(w_latent_out[:,:, np.newaxis], [1, 1, self.layers+1])
				feed_dict = {self.w_latent_in:w_latent_in, self.real_images_1:batch_images}
				gen_samples = session.run([self.output_gen], feed_dict=feed_dict)[0]
				write_sprite_image(filename=os.path.join(data_out_path, 'images/gen_samples_steady_epoch_%s.png' % epoch), data=gen_samples, metadata=False)

				# Image reconstruction from steady vector images.
				feed_dict = {self.real_images_2:gen_samples}
				enc_img = session.run([self.w_latent_e_out], feed_dict=feed_dict)[0]
				w_latent_in = np.tile(enc_img[:,:, np.newaxis], [1, 1, self.layers+1])
				feed_dict = {self.w_latent_in:w_latent_in, self.real_images_1:batch_images}
				gen_samples = session.run([self.output_gen], feed_dict=feed_dict)[0]
				write_sprite_image(filename=os.path.join(data_out_path, 'images/gen_samples_recon_epoch_%s.png' % epoch), data=gen_samples, metadata=False)
				