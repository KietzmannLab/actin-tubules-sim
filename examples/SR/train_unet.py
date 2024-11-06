import datetime
from csbdeep.io import load_training_data
from csbdeep.utils import axes_dict
from vollseg import ProjectionUpsamplingConfig, ProjectionUpsampling
import tensorflow as tf
from pathlib import Path

root_dir = '/Users/vkapoor/Downloads/Microtubules'
model_dir = Path(root_dir)/'SRModel'
Path(model_dir).mkdir(exist_ok=True)
train_data_file = f'{root_dir}/Train/SR/microtubule_sr_training_data.npz'
log_dir = "logs/fitSR/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
batch_size = 10
epochs = 10
unet_n_depth = 3
train_loss = 'mse'
unet_n_first = 48
unet_kern_size=3
train_epochs=400
train_batch_size=4
train_learning_rate=0.0001
upsampling_factor = 2

(X,Y), (X_val,Y_val), axes = load_training_data(train_data_file, validation_split=0.1, verbose=True)
total_data,  height, width, channels= X.shape
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

config = ProjectionUpsamplingConfig(axes, 
                                    n_channel_in, 
                                    n_channel_out, 
                                    unet_n_depth=unet_n_depth,
                                    train_loss=train_loss,
                                    unet_n_first=unet_n_first,
                                    unet_kern_size=unet_kern_size,
                                    train_epochs=epochs, 
                                    train_batch_size = batch_size, 
                                    train_learning_rate = train_learning_rate,
                                    upsampling_factor=upsampling_factor)

print(config)
vars(config)

model = ProjectionUpsampling(config, 'microtubule_unet', basedir=model_dir)
history = model.train(X,Y, validation_data=(X_val,Y_val))


def plot_history(history, *keys, save_path=None, **kwargs):
    """Plot (Keras) training history returned by :func:`CARE.train` and save the plot if save_path is provided."""
    import matplotlib.pyplot as plt

    logy = kwargs.pop('logy', False)

    if all((isinstance(k, str) for k in keys)):
        w, keys = 1, [keys]
    else:
        w = len(keys)

    plt.gcf()
    for i, group in enumerate(keys):
        plt.subplot(1, w, i + 1)
        for k in ([group] if isinstance(group, str) else group):
            plt.plot(history.epoch, history.history[k], '.-', label=k, **kwargs)
            if logy:
                plt.gca().set_yscale('log', nonposy='clip')
        plt.xlabel('epoch')
        plt.legend(loc='best')

    # plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        
plot_history(history, ['loss','val_loss'],['mse','val_mse','mae','val_mae'], save_path='training_history_plot.png')        