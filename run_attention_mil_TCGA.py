from models.mil.Attention_MIL_TCGA_LUAD import Attention_MIL
# Imports.
from data_manipulation.data import Data
import tensorflow as tf
import argparse
import random
import os 

# Folder permissions for cluster.
os.umask(0o002)
# H5 File bug over network file system.
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

parser = argparse.ArgumentParser(description='Deep Attention Multiple Instance Learning (MIL) trainer.')
parser.add_argument('--dataset',            dest='dataset',            type=str,   default='TCGAFFPE_set00_90pcFFPE',  help='Dataset to use.')
parser.add_argument('--z_dim',              dest='z_dim',              type=int,   default=200,                        help='Latent space size.')
parser.add_argument('--att_dim',            dest='att_dim',            type=int,   default=5,                          help='Dimensions of the attention network, default is 150.')
parser.add_argument('--bag_size',           dest='bag_size',           type=int,   default=10000,                      help='Maximum number of instaces for a bag, default is 10K.')
parser.add_argument('--img_size',           dest='img_size',           type=int,   default=224,                        help='Image size for the model.')
parser.add_argument('--epochs',             dest='epochs',             type=int,   default=50,                         help='Number epochs to run, default is 50 epochs.')
parser.add_argument('--learning_rate',      dest='learning_rate',      type=float, default=0.0001,                     help='Learning rate, default is 1e-4.')
parser.add_argument('--folds',              dest='folds',              type=int,   default=10,                         help='Number of random initializations.')
parser.add_argument('--model',              dest='model',              type=str,   default='Attention_MIL',            help='Attention MIL Model name.')
parser.add_argument('--gan_model',          dest='gan_model',          type=str,   default='PathologyGAN_Encoder',     help='Model name of latent projections.')
parser.add_argument('--hdf5_file_path',     dest='hdf5_file_path',     type=str,   default=None, required=True,        help='Path for latent representations H5 file, results on loss.csv.')
parser.add_argument('--hdf5_file_path_add', dest='hdf5_file_path_add', type=str,   default=None, required=False,       help='Path for additional latent representations H5 file, results on loss_2.csv.')
parser.add_argument('--gated',              dest='gated',              action='store_true', default=False,             help='Gated architecture for attention.')
args               = parser.parse_args()
dataset            = args.dataset
att_dim            = args.att_dim
z_dim              = args.z_dim
bag_size           = args.bag_size
epochs             = args.epochs
learning_rate      = args.learning_rate
folds              = args.folds
model              = args.model
gan_model          = args.gan_model
img_size           = args.img_size
hdf5_file_path     = args.hdf5_file_path
hdf5_file_path_add = args.hdf5_file_path_add
gated              = args.gated

if z_dim == 1024:
	h_latent = True
else:
	h_latent = False
	

# Dataset information.
main_path = os.path.dirname(os.path.realpath(__file__))
dbs_path = '/media/adalberto/Disk2/PhD_Workspace'
data_out_path = os.path.join(main_path, 'data_model_output')
data_out_path = os.path.join(data_out_path, model)
data_out_path = os.path.join(data_out_path, gan_model)
image_width = img_size
image_height = img_size
image_channels = 3
name_run = 'h%s_w%s_n%s_zdim%s_att%s_hlatent_%s_gated_%s' % (image_height, image_width, image_channels, z_dim, att_dim, h_latent, gated)
data_out_path = os.path.join(data_out_path, dataset)
data_out_path = os.path.join(data_out_path, name_run)

with tf.Graph().as_default():
    amil = Attention_MIL(z_dim=z_dim, att_dim=att_dim, bag_size=bag_size, learning_rate=learning_rate, use_gated=gated)
    amil.train(epochs=epochs, hdf5_file_path=hdf5_file_path, data_out_path=data_out_path, folds=folds, hdf5_file_path_add=hdf5_file_path_add, h_latent=h_latent)
