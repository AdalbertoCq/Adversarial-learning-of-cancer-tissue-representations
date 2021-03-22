# Imports.
from models.generative.gans.PathologyGAN_Encoder import PathologyGAN_Encoder
from data_manipulation.data import Data
import tensorflow as tf
import argparse
import os 

# Folder permissions for cluster.
os.umask(0o002)

parser = argparse.ArgumentParser(description='PathologyGAN Encoder trainer.')
parser.add_argument('--model',         dest='model',         type=str,            default='PathologyGAN_Encoder',   help='Model name.')
parser.add_argument('--img_size',      dest='img_size',      type=int,            default=224,                      help='Image size for the model.')
parser.add_argument('--img_ch',        dest='img_ch',        type=int,            default=3,                        help='Number of channels for the model.')
parser.add_argument('--dataset',       dest='dataset',       type=str,            default='vgh_nki',                help='Dataset to use.')
parser.add_argument('--marker',        dest='marker',        type=str,            default='he',                     help='Marker of dataset to use.')
parser.add_argument('--z_dim',         dest='z_dim',         type=int,            default=200,                      help='Latent space size.')
parser.add_argument('--epochs',        dest='epochs',        type=int,            default=45,                       help='Number epochs to run: default is 45 epochs.')
parser.add_argument('--batch_size',    dest='batch_size',    type=int,            default=64,                       help='Batch size, default size is 64.')
parser.add_argument('--check_every',   dest='check_every',   type=int,            default=10,                       help='Save checkpoint and generate samples every X epcohs.')
parser.add_argument('--restore',       dest='restore',       action='store_true', default=False,                    help='Restore previous run and continue.')
parser.add_argument('--report',        dest='report',        action='store_true', default=False,                    help='Report latent space figures.')
parser.add_argument('--main_path',     dest='main_path',     type=str,            default=None,                     help='Path for the output run.')
parser.add_argument('--dbs_path',      dest='dbs_path',      type=str,            default=None,                     help='Directory with DBs to use.')
args             = parser.parse_args()
model            = args.model
image_width      = args.img_size
image_height     = args.img_size
image_channels   = args.img_ch
dataset          = args.dataset
marker           = args.marker
z_dim            = args.z_dim
epochs           = args.epochs
batch_size       = args.batch_size
main_path        = args.main_path
dbs_path         = args.dbs_path
restore          = args.restore
report           = args.report
check_every      = args.check_every

# Main paths for data output and databases.
if main_path is None:
	main_path = os.path.dirname(os.path.realpath(__file__))
if dbs_path is None:
	dbs_path = os.path.dirname(os.path.realpath(__file__))

# Dataset information.
name_run      = 'h%s_w%s_n%s_zdim%s' % (image_height, image_width, image_channels, z_dim)
data_out_path = os.path.join(main_path, 'data_model_output')
data_out_path = os.path.join(data_out_path, model)
data_out_path = os.path.join(data_out_path, dataset)
data_out_path = os.path.join(data_out_path, name_run)

# Hyperparameters for training.
learning_rate_g   = 1e-4
learning_rate_d   = 1e-4   
learning_rate_e   = 1e-4
beta_1            = 0.5
beta_2            = 0.9
regularizer_scale = 1e-4
style_mixing      = 0.0

# Model Architecture param.
layers_map    = {448:6, 224:5, 112:4, 56:3, 28:2}
layers        = layers_map[image_height]
noise_input_f = True
spectral      = True
attention     = 28
alpha         = 0.2
n_critic      = 5
gp_coeff      = .65
use_bn        = False
init          = 'orthogonal'
loss_type     = 'relativistic gradient penalty'

# Collect dataset.	
data = Data(dataset=dataset, marker=marker, patch_h=image_height, patch_w=image_width, n_channels=image_channels, batch_size=batch_size, project_path=dbs_path)

# Run PathologyGAN Encoder.
with tf.Graph().as_default():
	# Instantiate Model.
    pathgan_encoder = PathologyGAN_Encoder(data=data, z_dim=z_dim, layers=layers, use_bn=use_bn, alpha=alpha, beta_1=beta_1, init=init, regularizer_scale=regularizer_scale, 
    									   style_mixing=style_mixing, attention=attention, spectral=spectral, noise_input_f=noise_input_f, learning_rate_g=learning_rate_g, 
    									   learning_rate_d=learning_rate_d, learning_rate_e=learning_rate_e, beta_2=beta_2, n_critic=n_critic, gp_coeff=gp_coeff, 
    									   loss_type=loss_type, model_name=model)
   	# Training.
    losses = pathgan_encoder.train(epochs=epochs, data_out_path=data_out_path, data=data, restore=restore, print_epochs=10, checkpoint_every=check_every, report=report)


