from models.generative.gans.PathologyGAN_Encoder   import PathologyGAN_Encoder as PathologyGAN
from models.evaluation.features import *
from data_manipulation.data import Data
import tensorflow as tf
import argparse
import os 
os.umask(0o002)

parser = argparse.ArgumentParser(description='PathologyGAN fake image generator and feature extraction.')
parser.add_argument('--checkpoint',    dest='checkpoint',    required= True,      help='Path to pre-trained weights (.ckt) of PathologyGAN.')
parser.add_argument('--real_hdf5',     dest='real_hdf5',     type=str,            default=200, required=True, help='Path for real image to encode.')
parser.add_argument('--batch_size',    dest='batch_size',    type=int,            default=50,                 help='Batch size.')
parser.add_argument('--z_dim',         dest='z_dim',         type=int,            default=200,                help='Latent space size.')
parser.add_argument('--model',         dest='model',         type=str,            default='PathologyGAN',     help='Model name.')
parser.add_argument('--img_size',      dest='img_size',      type=int,            default=224,                help='Image size for the model.')
parser.add_argument('--img_ch',        dest='img_ch',        type=int,            default=3,                  help='Image size for the model.')
parser.add_argument('--dataset',       dest='dataset',       type=str,            default='vgh_nki',          help='Dataset to use.')
parser.add_argument('--marker',        dest='marker',        type=str,            default='he',               help='Marker of dataset to use.')
parser.add_argument('--dbs_path',      dest='dbs_path',      type=str,            default=None,               help='Directory with DBs to use.')
parser.add_argument('--main_path',     dest='main_path',     type=str,            default=None,               help='Path for the output run.')
parser.add_argument('--num_clusters',  dest='num_clusters',  type=int,            default=100,                help='Number of clusters for PathologyGAN_plus.')
parser.add_argument('--clust_percent', dest='clust_percent', type=float,          default=1.0,                help='Percentage of the original data to consider on clustering.')   
parser.add_argument('--features',      dest='features',      action='store_true', default=False,              help='Flag to run features over the images.')
parser.add_argument('--save_img',      dest='save_img',      action='store_true', default=False,              help='Save reconstructed images in the H5 file.')
args = parser.parse_args()
args = parser.parse_args()
checkpoint     = args.checkpoint
batch_size     = args.batch_size
z_dim          = args.z_dim
model          = args.model
real_hdf5      = args.real_hdf5
image_width    = args.img_size
image_height   = args.img_size
image_channels = args.img_ch
dataset        = args.dataset
marker         = args.marker
dbs_path       = args.dbs_path
main_path      = args.main_path
num_clusters   = args.num_clusters
clust_percent  = args.clust_percent
features       = args.features
save_img       = args.save_img

# Paths for runs and datasets.
if dbs_path is None:
    dbs_path  = os.path.dirname(os.path.realpath(__file__))
if main_path is None:
    main_path = os.path.dirname(os.path.realpath(__file__))

# Hyperparameters for training.
learning_rate_g   = 1e-4
learning_rate_d   = 1e-4
learning_rate_e   = 1e-4
regularizer_scale = 1e-4
beta_1            = 0.5
beta_2            = 0.9
style_mixing      = 0.5

# Model Architecture param.
layers_map    = {224:5, 112:4, 56:3, 28:2}
layers        = layers_map[image_height]
alpha         = 0.2
n_critic      = 5
gp_coeff      = .65
use_bn        = False
init          = 'orthogonal'
loss_type     = 'relativistic gradient penalty'
noise_input_f = True
spectral      = True
attention     = 28


data = Data(dataset=dataset, marker=marker, patch_h=image_height, patch_w=image_width, n_channels=image_channels, batch_size=batch_size, project_path=dbs_path, labels=None)

# Instantiate and project images.
with tf.Graph().as_default():
	# Instantiate Model.
	pathgan = PathologyGAN(data=data, z_dim=z_dim, layers=layers, use_bn=use_bn, alpha=alpha, beta_1=beta_1, init=init, regularizer_scale=regularizer_scale, 
						   style_mixing=style_mixing, attention=attention, spectral=spectral, noise_input_f=noise_input_f, learning_rate_g=learning_rate_g, 
						   learning_rate_d=learning_rate_d, learning_rate_e=learning_rate_e, beta_2=beta_2, n_critic=n_critic, gp_coeff=gp_coeff, 
						   loss_type=loss_type, model_name=model)
	real_hdf5_path, num_samples = real_encode_from_checkpoint(model=pathgan, data=data, data_out_path=main_path, checkpoint=checkpoint, real_hdf5=real_hdf5, batches=batch_size, save_img=save_img)
