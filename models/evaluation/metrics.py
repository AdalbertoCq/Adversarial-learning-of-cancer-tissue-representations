from sklearn.metrics import *
import numpy as np
import random
import shutil
import h5py
import os


# Compute different metrics for given set: Accuracy, Recall, Precision, and AUC.
def compute_metrics_attention(model, session, slides, patterns, latent, subset_slides=None, labels=None, return_weights=False, top_perc=0.001):
	# Variables to return.
	# Relevant information related to outcomes.
	relevant_indeces = list()
	relevant_labels  = list()
	relevant_patches = list()
	relevant_slides  = list()
	relevant_weights = list()
	# Prediction, True labels for metrics.
	prob_set         = list()
	pred_set         = list()
	class_set         = list()


	# Unique slides to iterate through.
	if subset_slides is None:
		unique_slides    = list(np.unique(slides))
	# Use specified slides: Histopology subtypes.
	else:
		unique_slides    = subset_slides

	# Iterate through slides.
	i = 0
	for slide in unique_slides:
		# Gather tiles for the slide.
		indxs = np.argwhere(slides[:]==slide)[:,0]
		random.shuffle(indxs)
		# print('Indxs', indxs.shape)
		indxs = np.array(sorted(indxs[:model.bag_size]))
		latents_batch = latent[indxs, :]

		# Label processing for the tile.
		if model.mult_class == 21:
			label_batch_int = labels[indxs[0]]
		else:
			label_instances = patterns[indxs[0]]
			label_batch_int = model.process_label(label_instances)
		label_batch = model.one_hot_encoder.transform([[label_batch_int]])
		
		# Run the model.
		feed_dict = {model.represenation_input:latents_batch}
		if return_weights:
			prob_batch, weights = session.run([model.prob, model.weights], feed_dict=feed_dict)
			ind = np.argsort(weights.reshape((1,-1)))[0,:]
		else:
			prob_batch = session.run([model.prob], feed_dict=feed_dict)[0]

		# Keep track of outcomes for slide.
		prob_set.append(prob_batch)
		pred_set.append(np.argmax(prob_batch))
		class_set.append(label_batch_int)

		# Keep relevant tiles for the outcome.
		if return_weights and (pred_set[i]==class_set[i]):
			top_patches = int(ind.shape[0]*top_perc)
			if top_patches == 0: top_patches += 1

			latents_sample = latents_batch[ind[-top_patches:]]
			indeces_sample = indxs[ind[-top_patches:]]
			labels_sample  = np.ones((latents_sample.shape[0],1))*pred_set[i]
			slide_sample   = [slide]*latents_sample.shape[0]
			weights_sample = weights[ind[-top_patches:]]

			relevant_patches.append(latents_sample)
			relevant_slides.extend(slide_sample)
			relevant_labels.append(labels_sample.reshape((-1,1)))
			relevant_indeces.append(indeces_sample.reshape((-1,1)))
			relevant_weights.append(weights_sample.reshape((-1,1)))
		i += 1

	# Reshape into np.array
	prob_set  = np.vstack(prob_set).reshape((-1,model.mult_class))
	pred_set  = np.vstack(pred_set)
	class_set = np.vstack(class_set)

	# Accuracy and Confusion Matrix.
	cm             = confusion_matrix(y_true=class_set, y_pred=pred_set)
	cm             = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	acc_per_class  = cm.diagonal()
	accuracy_total = balanced_accuracy_score(y_true=class_set,  y_pred=pred_set)
	accuracy       = np.round([accuracy_total] + list(acc_per_class), 2)
	# Recall 
	recall         = np.round(recall_score(y_true=class_set,    y_pred=pred_set, average=None), 2)
	# Precision
	precision      = np.round(precision_score(y_true=class_set, y_pred=pred_set, average=None), 2)
	# AUC.
	try:
		auc_per_class  = roc_auc_score(y_true=model.one_hot_encoder.transform(class_set.reshape((-1,1))), y_score=prob_set, average=None)
		# Macro, subject to data imbalance.
		auc_total      = np.mean(auc_per_class)
		auc_all        = np.round([auc_total] + list(auc_per_class), 2)
	except:
		auc_all = [None]
		for class_ in range(model.mult_class):
			try:
			    fpr, tpr, thresholds = roc_curve(model.one_hot_encoder.transform(class_set.reshape((-1,1)))[:, class_], prob_set[:, class_])
			    roc_auc = np.round(auc(fpr, tpr), 2)
			except:
				roc_auc = None
			auc_all.append(roc_auc)
		

	# In case we need the relevant tiles.
	if not return_weights:
		return accuracy, recall, precision, auc_all, class_set, pred_set, prob_set
	else:
		relevant_patches = np.vstack(relevant_patches)
		relevant_labels = np.vstack(relevant_labels)
		relevant_indeces = np.vstack(relevant_indeces)
		relevant_weights = np.vstack(relevant_weights)
		return [accuracy, recall, precision, auc_all], [relevant_patches, relevant_labels, relevant_indeces, relevant_slides, relevant_weights], [class_set, pred_set, prob_set]

# Compute different metrics for given set: Accuracy, Recall, Precision, and AUC.
def compute_metrics_attention_multimag(model, session, slides, patterns, latent_20x, latent_5x, subset_slides=None, labels=None):

	# Prediction, True labels for metrics.
	prob_set         = list()
	pred_set         = list()
	class_set         = list()


	# Unique slides to iterate through.
	if subset_slides is None:
		unique_slides    = list(np.unique(slides))
	# Use specified slides: Histopology subtypes.
	else:
		unique_slides    = subset_slides

	# Iterate through slides.
	i = 0
	for slide in unique_slides:
		if slide == '': continue

		# Gather tiles for the slide.
		indxs = np.argwhere(slides[:]==slide)[:,0]
		start_ind = sorted(indxs)[0]
		num_tiles_5x = indxs.shape[0]

		# Slide latents for 20x and 5x.
		# random.shuffle(indxs)
		# indxs = sorted(indxs[:model.bag_size])
		lantents_5x_batch  = latent_5x[start_ind:start_ind+num_tiles_5x]
		lantents_20x_batch = latent_20x[start_ind:start_ind+num_tiles_5x]

		# Label processing for the tile.
		label_instances = patterns[start_ind]
		label_batch_int = model.process_label(label_instances[0])
		# label_batch = model.one_hot_encoder.transform([[label_batch]])

		# Run the model.
		feed_dict = {model.represenation_input_20x:lantents_20x_batch, model.represenation_input_5x:lantents_5x_batch}
		prob_batch = session.run([model.prob], feed_dict=feed_dict)[0]

		# Keep track of outcomes for slide.
		prob_set.append(prob_batch)
		pred_set.append(np.argmax(prob_batch))
		class_set.append(label_batch_int)

		i += 1

	# Reshape into np.array
	prob_set  = np.vstack(prob_set).reshape((-1,model.mult_class))
	pred_set  = np.vstack(pred_set)
	class_set = np.vstack(class_set)

	# Accuracy and Confusion Matrix.
	cm             = confusion_matrix(y_true=class_set, y_pred=pred_set)
	cm             = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	acc_per_class  = cm.diagonal()
	accuracy_total = balanced_accuracy_score(y_true=class_set,  y_pred=pred_set)
	accuracy       = np.round([accuracy_total] + list(acc_per_class), 2)
	# Recall 
	recall         = np.round(recall_score(y_true=class_set,    y_pred=pred_set, average=None), 2)
	# Precision
	precision      = np.round(precision_score(y_true=class_set, y_pred=pred_set, average=None), 2)
	# AUC.
	try:
		auc_per_class  = roc_auc_score(y_true=model.one_hot_encoder.transform(class_set.reshape((-1,1))), y_score=prob_set, average=None)
		# Macro, subject to data imbalance.
		auc_total      = np.mean(auc_per_class)
		auc_all        = np.round([auc_total] + list(auc_per_class), 2)
	except:
		auc_all = [None]
		for class_ in range(model.mult_class):
			try:
			    fpr, tpr, thresholds = roc_curve(model.one_hot_encoder.transform(class_set.reshape((-1,1)))[:, class_], prob_set[:, class_])
			    roc_auc = np.round(auc(fpr, tpr), 2)
			except:
				roc_auc = None
			auc_all.append(roc_auc)
		

	# In case we need the relevant tiles.
	return accuracy, recall, precision, auc_all, class_set, pred_set, prob_set


# Compute different metrics for given set: Accuracy, Recall, Precision, and AUC.
def compute_metrics_attention_multimagnifications(model, session, slides, patterns, latent_20x, latent_10x, latent_5x, subset_slides=None, labels=None):

	# Prediction, True labels for metrics.
	prob_set         = list()
	pred_set         = list()
	class_set         = list()


	# Unique slides to iterate through.
	if subset_slides is None:
		unique_slides    = list(np.unique(slides))
	# Use specified slides: Histopology subtypes.
	else:
		unique_slides    = subset_slides

	# Iterate through slides.
	i = 0
	for slide in unique_slides:
		if slide == '': continue

		# Gather tiles for the slide.
		indxs = np.argwhere(slides[:]==slide)[:,0]
		start_ind = sorted(indxs)[0]
		num_tiles_5x = indxs.shape[0]

		# Slide latents for 20x and 5x.
		# random.shuffle(indxs)
		# indxs = sorted(indxs[:model.bag_size])
		lantents_5x_batch  = latent_5x[start_ind:start_ind+num_tiles_5x]
		lantents_10x_batch = latent_10x[start_ind:start_ind+num_tiles_5x]
		lantents_20x_batch = latent_20x[start_ind:start_ind+num_tiles_5x]

		# Label processing for the tile.
		label_instances = patterns[start_ind]
		label_batch_int = model.process_label(label_instances[0])
		# label_batch = model.one_hot_encoder.transform([[label_batch]])

		# Run the model.
		feed_dict = {model.represenation_input_20x:lantents_20x_batch, model.represenation_input_10x:lantents_10x_batch, model.represenation_input_5x:lantents_5x_batch}
		prob_batch = session.run([model.prob], feed_dict=feed_dict)[0]

		# Keep track of outcomes for slide.
		prob_set.append(prob_batch)
		pred_set.append(np.argmax(prob_batch))
		class_set.append(label_batch_int)

		i += 1

	# Reshape into np.array
	prob_set  = np.vstack(prob_set).reshape((-1,model.mult_class))
	pred_set  = np.vstack(pred_set)
	class_set = np.vstack(class_set)

	# Accuracy and Confusion Matrix.
	cm             = confusion_matrix(y_true=class_set, y_pred=pred_set)
	cm             = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	acc_per_class  = cm.diagonal()
	accuracy_total = balanced_accuracy_score(y_true=class_set,  y_pred=pred_set)
	accuracy       = np.round([accuracy_total] + list(acc_per_class), 2)
	# Recall 
	recall         = np.round(recall_score(y_true=class_set,    y_pred=pred_set, average=None), 2)
	# Precision
	precision      = np.round(precision_score(y_true=class_set, y_pred=pred_set, average=None), 2)
	# AUC.
	try:
		auc_per_class  = roc_auc_score(y_true=model.one_hot_encoder.transform(class_set.reshape((-1,1))), y_score=prob_set, average=None)
		# Macro, subject to data imbalance.
		auc_total      = np.mean(auc_per_class)
		auc_all        = np.round([auc_total] + list(auc_per_class), 2)
	except:
		auc_all = [None]
		for class_ in range(model.mult_class):
			try:
			    fpr, tpr, thresholds = roc_curve(model.one_hot_encoder.transform(class_set.reshape((-1,1)))[:, class_], prob_set[:, class_])
			    roc_auc = np.round(auc(fpr, tpr), 2)
			except:
				roc_auc = None
			auc_all.append(roc_auc)
		

	# In case we need the relevant tiles.
	return accuracy, recall, precision, auc_all, class_set, pred_set, prob_set
	

# Compute different metrics for given set: Accuracy, Recall, Precision, and AUC.
def save_weights_attention(model, set_type, session, output_path, slides, patterns, latent, subset_slides=None, labels=None):
	
	# Unique slides to iterate through.
	if subset_slides is None:
		unique_slides    = list(np.unique(slides))
	# Use specified slides: Histopology subtypes.
	else:
		unique_slides    = subset_slides

	# Variables to return.
	weights_set   = np.zeros((latent.shape[0], 1))
	probabilities = np.zeros((slides.shape[0], 2))

	# Iterate through slides.
	for slide in unique_slides:
		# Gather tiles for the slide.
		indxs = np.argwhere(slides[:]==slide)[:,0]
		random.shuffle(indxs)
		# print('Indxs', indxs.shape)
		indxs = np.array(sorted(indxs[:model.bag_size]))
		latents_batch = latent[indxs, :]

		# Run the model.
		feed_dict = {model.represenation_input:latents_batch}
		prob_batch, weights = session.run([model.prob, model.weights], feed_dict=feed_dict)

		for i, index in enumerate(indxs):
			weights_set[index]   = weights[i,0]
			probabilities[index] = prob_batch
	
	# Store weights in H5 file.
	hdf5_path = os.path.join(output_path, 'hdf5_attention_weights_%s.h5' % set_type)
	with h5py.File(hdf5_path, mode='w') as hdf5_content:   
	    weight_storage = hdf5_content.create_dataset(name='weights',       shape=weights_set.shape,   dtype=weights_set.dtype)
	    probab_storage = hdf5_content.create_dataset(name='probabilities', shape=probabilities.shape, dtype=weights_set.dtype)
	    for i in range(weights_set.shape[0]):
	        weight_storage[i]    = weights_set[i]
	        probab_storage[i]    = probabilities[i]

# Compute different metrics for given set: Accuracy, Recall, Precision, and AUC.
def save_weights_attention_multimag(model, set_type, session, output_path, slides, patterns, latent_20x, latent_5x, subset_slides=None, labels=None):
	
	# Unique slides to iterate through.
	if subset_slides is None:
		unique_slides    = list(np.unique(slides))
	# Use specified slides: Histopology subtypes.
	else:
		unique_slides    = subset_slides

	# Variables to return.
	weights_5x_set  = np.zeros((latent_5x.shape[0], 1))
	weights_20x_set = np.zeros((latent_5x.shape[0], 16,1))
	probabilities   = np.zeros((latent_5x.shape[0], 2))

	# Iterate through slides.
	for slide in unique_slides:
		if slide == '': continue

		# Gather tiles for the slide.
		indxs = np.argwhere(slides[:]==slide)[:,0]
		start_ind = sorted(indxs)[0]
		num_tiles_5x = indxs.shape[0]

		# Slide latents for 20x and 5x.
		# random.shuffle(indxs)
		# indxs = sorted(indxs[:model.bag_size])
		lantents_5x_batch  = latent_5x[start_ind:start_ind+num_tiles_5x]
		lantents_20x_batch = latent_20x[start_ind:start_ind+num_tiles_5x]

		# Label processing for the tile.
		label_instances = patterns[start_ind]
		label_batch_int = model.process_label(label_instances[0])

		# Run the model.
		feed_dict = {model.represenation_input_20x:lantents_20x_batch, model.represenation_input_5x:lantents_5x_batch}
		prob_batch, weights_5x, weights_20x = session.run([model.prob, model.weights, model.weights_20x], feed_dict=feed_dict)

		for i, index in enumerate(indxs):
			weights_5x_set[index]  = weights_5x[i,0]
			weights_20x_set[index, :, 0] = weights_20x[i,:,0]
			probabilities[index] = prob_batch
	
	# Store weights in H5 file.
	hdf5_path = os.path.join(output_path, 'hdf5_attention_weights_%s.h5' % set_type)
	with h5py.File(hdf5_path, mode='w') as hdf5_content:   
	    weight_20x_storage = hdf5_content.create_dataset(name='weights_20x',   shape=weights_20x_set.shape, dtype=weights_20x_set.dtype)
	    weight_5x_storage  = hdf5_content.create_dataset(name='weights_5x',    shape=weights_5x_set.shape,  dtype=weights_5x_set.dtype)
	    probab_storage     = hdf5_content.create_dataset(name='probabilities', shape=probabilities.shape,   dtype=weights_5x_set.dtype)
	    for i in range(weights_5x_set.shape[0]):
	        weight_5x_storage[i]  = weights_5x_set[i]
	        weight_20x_storage[i] = weights_20x_set[i]
	        probab_storage[i]     = probabilities[i]


# Compute different metrics for given set: Accuracy, Recall, Precision, and AUC.
def save_weights_attention_multimagnifications(model, set_type, session, output_path, slides, patterns, latent_20x, latent_10x, latent_5x, subset_slides=None, labels=None):
	
	# Unique slides to iterate through.
	if subset_slides is None:
		unique_slides    = list(np.unique(slides))
	# Use specified slides: Histopology subtypes.
	else:
		unique_slides    = subset_slides

	# Variables to return.
	weights_5x_set  = np.zeros((latent_5x.shape[0], 1))
	weights_10x_set = np.zeros((latent_5x.shape[0], 4, 1))
	weights_20x_set = np.zeros((latent_5x.shape[0], 4, 4, 1))
	probabilities   = np.zeros((latent_5x.shape[0], 2))

	# Iterate through slides.
	for slide in unique_slides:
		if slide == '': continue

		# Gather tiles for the slide.
		indxs = np.argwhere(slides[:]==slide)[:,0]
		start_ind = sorted(indxs)[0]
		num_tiles_5x = indxs.shape[0]

		# Slide latents for 20x and 5x.
		# random.shuffle(indxs)
		# indxs = sorted(indxs[:model.bag_size])
		lantents_5x_batch  = latent_5x[start_ind:start_ind+num_tiles_5x]
		lantents_10x_batch = latent_10x[start_ind:start_ind+num_tiles_5x]
		lantents_20x_batch = latent_20x[start_ind:start_ind+num_tiles_5x]

		# Label processing for the tile.
		label_instances = patterns[start_ind]
		label_batch_int = model.process_label(label_instances[0])

		# Run the model.
		feed_dict = {model.represenation_input_20x:lantents_20x_batch, model.represenation_input_10x:lantents_10x_batch, model.represenation_input_5x:lantents_5x_batch}
		prob_batch, weights_5x, weights_10x, weights_20x = session.run([model.prob, model.weights, model.weights_10x, model.weights_20x], feed_dict=feed_dict)

		for i, index in enumerate(indxs):
			weights_5x_set[index]  = weights_5x[i,0]
			weights_10x_set[index, :, 0] = weights_10x[i,:,0]
			weights_20x_set[index, :, :, 0] = weights_20x[i,:,:,0]
			probabilities[index] = prob_batch
	
	# Store weights in H5 file.
	hdf5_path = os.path.join(output_path, 'hdf5_attention_weights_%s.h5' % set_type)
	with h5py.File(hdf5_path, mode='w') as hdf5_content:   
	    weight_20x_storage = hdf5_content.create_dataset(name='weights_20x',   shape=weights_20x_set.shape, dtype=weights_20x_set.dtype)
	    weight_10x_storage = hdf5_content.create_dataset(name='weights_10x',   shape=weights_10x_set.shape, dtype=weights_10x_set.dtype)
	    weight_5x_storage  = hdf5_content.create_dataset(name='weights_5x',    shape=weights_5x_set.shape,  dtype=weights_5x_set.dtype)
	    probab_storage     = hdf5_content.create_dataset(name='probabilities', shape=probabilities.shape,   dtype=weights_5x_set.dtype)
	    for i in range(weights_5x_set.shape[0]):
	        weight_5x_storage[i]  = weights_5x_set[i]
	        weight_10x_storage[i] = weights_10x_set[i]
	        weight_20x_storage[i] = weights_20x_set[i]
	        probab_storage[i]     = probabilities[i]
