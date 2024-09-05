import datetime
from csbdeep.io import load_training_data
from csbdeep.utils import axes_dict, plot_some, plot_history
import matplotlib.pyplot as plt
from actin_tubules_sim.models import Denoiser, Train_RDL_Denoising
from actin_tubules_sim.loss import mse_ssim
import tensorflow as tf
import os
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from tensorflow.keras.models import load_model


root_dir = '/Users/vkapoor/Downloads/Microtubules'
den_model_dir = Path(root_dir)/'DenoisingCARE'
sr_model_dir = Path(root_dir)/'SRModel'
Path(den_model_dir).mkdir(exist_ok=True)
Path(sr_model_dir).mkdir(exist_ok=True)
train_data_file = f'{root_dir}/Train/DN/microtubule_dn_training_data.npz'
log_dir = "logs/fitDN/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


(X,Y), (X_val,Y_val), axes = load_training_data(train_data_file, validation_split=0.1, verbose=True)

c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

X = tf.squeeze(X, axis=-1)
X_val = tf.squeeze(X_val, axis=-1)
Y = tf.squeeze(Y, axis=-1)
Y_val = tf.squeeze(Y_val, axis=-1)
X = tf.transpose(X, perm=[0, 2, 3, 1])
X_val = tf.transpose(X_val, perm=[0, 2, 3, 1])
Y = tf.transpose(Y, perm=[0, 2, 3, 1])
Y_val = tf.transpose(Y_val, perm=[0, 2, 3, 1])

print(X.shape,Y.shape,X_val.shape)

data = (X, Y)
data_val = (X_val, Y_val)

init_lr = 1e-4
batch_size = 3
epochs = 10
beta_1=0.9
beta_2=0.999
wavelength = 0.488 
excNA = 1.35
dx = 62.6e-3
dy = dx
dxy = dx 
scale_gt = 2.0
setupNUM = 1
space = wavelength/excNA
k0mod = 1 / space
napodize = 10
nphases = 3
ndirs = 3
sigma_x = 0.5
sigma_y = 0.5
recalcarrays = 2
ifshowmodamp = 0
norders = int((nphases + 1) / 2)
if setupNUM == 0:
     k0angle_c = [1.48, 2.5272, 3.5744]
     k0angle_g = [0.0908, -0.9564, -2.0036]  
if setupNUM == 1:
     k0angle_c = [-1.66, -0.6128, 0.4344]
     k0angle_g = [3.2269, 2.1797, 1.1325]      
if setupNUM == 2:
     k0angle_c = [1.5708, 2.618, 3.6652]
     k0angle_g = [0, -1.0472, -2.0944] 
total_data,  height, width, channels = X.shape


parameters = {
    'Ny': height,
    'Nx': width,
    'wavelength':wavelength,
    'excNA':excNA,
    'ndirs':ndirs,
    'nphases':nphases,
    'init_lr': init_lr,
    'ifshowmodamp':ifshowmodamp,
    'batch_size': batch_size,
    'epochs': epochs,
    'beta_1':beta_1,
    'beta_2':beta_2,
    'scale_gt': scale_gt,
    'setupNUM': setupNUM,
    'k0angle_c':k0angle_c,
    'k0angle_g':k0angle_g,
    'recalcarrays':recalcarrays,
    'dxy':dxy,
    'space':space,
    'k0mod':k0mod,
    'norders':norders,
    'napodize':napodize,
    'scale': scale_gt,
    'sigma_x': sigma_x,
    'sigma_y': sigma_y,
    'log_dir': log_dir,
    'den_model_dir': den_model_dir,
    'sr_model_dir': sr_model_dir
    
}

if len(os.listdir(sr_model_dir)) > 0:

  with tf.keras.utils.custom_object_scope({'mse_ssim': mse_ssim}):
    if len(os.listdir(sr_model_dir)) > 0:
        print(f'Loading model from {sr_model_dir}')
        Trainingmodel_dfcan = load_model(sr_model_dir)
else:
  assert 'DFCAN model has to be trained before training RDL denosier'     
  
 
 
Trainingmodel_denoise = Denoiser((height, width, nphases))
optimizer = Adam(learning_rate=init_lr, beta_1=beta_1, beta_2=beta_2)
Trainingmodel_denoise.compile(loss=mse_ssim, optimizer=optimizer)
Trainingmodel_denoise.summary()

tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
lrate = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1)
hrate = callbacks.History() 

rdl_denoising = Train_RDL_Denoising(
                    srmodel=Trainingmodel_dfcan, 
                    denmodel=Trainingmodel_denoise,
                    loss_fn=mse_ssim,
                    optimizer=optimizer,
                    parameters = parameters)

rdl_denoising.fit(data= data, data_val = data_val)