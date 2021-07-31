import importlib.util
spec = importlib.util.spec_from_file_location("help_train", "/data/a/cpac/aurora/MDN_phoZ/help_train.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
import tensorflow.keras as keras
import tensorflow_probability as tfp
tfd = tfp.distributions
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.python import tf2 # Activate TF2 behavior:
if not tf2.enabled():
    import tensorflow.compat.v2 as tf
    tf.enable_v2_behavior()
    assert tf2.enabled()

def custom_loss(layer):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, mu, var, pi):
        mixture_distribution = tfp.distributions.Categorical(probs=pi)
        distribution = tfp.distributions.Normal(loc=mu, scale=var)
        likelihood = tfp.distributions.MixtureSameFamily(mixture_distribution=mixture_distribution,components_distribution=distribution)

        log_likelihood = -1.0*likelihood.log_prob(tf.transpose(y_true)) # A little confusing (talk later)
        mean_loss = tf.reduce_mean(log_likelihood)

        return mean_loss
    return loss

def run(point):
    global HISTORY
    X_train, y_train = pickle.load(open( b"/data/a/cpac/aurora/MDN_phoZ/training_data.obj", "rb") )
    X_test, y_test = pickle.load(open( b"/data/a/cpac/aurora/MDN_phoZ/validation_data.obj", "rb") )
    D = X_train.shape[1]
    K = 3 # number of mixture components
        
    y_true = tf.keras.Input(shape=(1,))
    inputs = tf.keras.Input(shape=(D,))
    
    non_lin_act = tf.nn.relu #tf.nn.tanh
    y_true = tf.keras.Input(shape=(1,))
    inputs = tf.keras.Input(shape=(D,))
    layer_1 = tf.keras.layers.Dense(units=point['units'], activation=point['activation'])(inputs)
    layer_1a = tf.keras.layers.Dense(units=point['units'], activation=point['activation'])(layer_1)
    layer_1b = tf.keras.layers.Dense(units=point['units'], activation=point['activation'])(layer_1a)
    layer_1c = tf.keras.layers.Dense(units=point['units'], activation=point['activation'])(layer_1b)
    layer_2 = tf.keras.layers.Dense(units=point['units'], activation=point['activation'])(layer_1c)
    layer_3 = tf.keras.layers.Dense(units=point['units'], activation=point['activation'])(layer_2)
    layer_4 = tf.keras.layers.Dense(units=point['units'], activation=point['activation'])(layer_3)
    layer_5 = tf.keras.layers.Dense(units=point['units'], activation=point['activation'])(layer_4)
    layer_6 = tf.keras.layers.Dense(units=point['units'], activation=point['activation'])(layer_5)
    
    # Gaussian mixture model
    # Should I consider changing the hyperparameters in here? i.e. activation function, or units K?
    mu = tf.keras.layers.Dense(units=K, activation=None, name="mu")(layer_6)
    var = tf.keras.backend.exp(tf.keras.layers.Dense(units=K, activation=tf.nn.softplus, name="sigma")(layer_6))
    pi = tf.keras.layers.Dense(units=K, activation=tf.nn.softmax, name="mixing")(layer_6)

    model_train = Model([inputs, y_true], [mu, var, pi], name='mdn')
    
    model_train.add_loss(custom_loss(inputs)(y_true, mu, var, pi))
    model_train.compile(optimizer='Nadam') # do we still need an optimizer, now that we're using deephyper?
    model_train.summary()
    
    def decay(epoch): # I still don't actually know what epoch is?
        if (epoch < 1):
            return point['lr']
        else:
            return point['lr']*(1.0/(1.0+point['lr']*(epoch)))
    
    class PrintLR(tf.keras.callbacks.Callback): # Print learning rate at every epoch
        def on_epoch_end(self, epoch, logs=None):
            print('\nLearning rate for epoch {} is {}'.format(epoch + 1, model_train.optimizer.lr.numpy()))

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        tf.keras.callbacks.LearningRateScheduler(decay),
        PrintLR()
    ]
    
    history = model_train.fit([X_train, y_train], 
                              validation_data=(X_test, y_test), 
                              validation_split = 0.1, # not sure what this is?
                              epochs=point['nepochs'],
                              batch_size = point['batch_size'], 
                              callbacks=callbacks)
    # uncomment if you want to save training. Leave commented for deephyper training (to save space)
    #save_mod = '/data/a/cpac/aurora/MDN_phoZ/saved_hubs/tf2models/'+'Train_'+Trainset+'_lr_'+str(point['lr'])+'_dr'+str(point['dr'])+'_ne'+str(point['nepochs'])+'_k'+str(K)+'_nt'+ '_activation_' + str(point['activation']) + "_batch_size_" + str(point['batch_size'])
    #model_train.save_weights(save_mod + '.h5')

    HISTORY = history.history # don't know what this is?

    return history.history['loss'][-1], history.history['val_loss'][-1]