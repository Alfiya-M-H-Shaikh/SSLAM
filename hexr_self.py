# Necessary packages
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend

from utils import mask_generator, pretext_generator

#%%
"""
The log cosh reconstruction loss function
"""
def log_cosh(y_true, y_pred):
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.cast(y_true, y_pred.dtype)
  def _logcosh(x):
    return x + tf.math.softplus(-2. * x) - tf.cast(tf.math.log(2.), x.dtype)
  return backend.mean(_logcosh(y_pred - y_true), axis=-1)

#%%
"""
Parameterized Elliot Activation Function
"""
initializer0 = keras.initializers.RandomUniform(minval = -1, maxval =1)
initializer1 = keras.initializers.RandomUniform(minval = 0.5, maxval =3)

def param_elliot_function( signal, k1, k2 ,  derivative=False ):
    s = 1 # steepness
    
    abs_signal = (1 + tf.math.abs(signal * s))
    if derivative:
        return 0.5 * s / abs_signal**2
    else:
        # Return the activation signal
        return (k1*(signal * s) / abs_signal + k2)

class ParamElliotfn(keras.layers.Layer):
    def __init__(self, trainable = True):
        super(ParamElliotfn, self).__init__()
        self.k1 = self.add_weight(name='k', shape = (), initializer=initializer0, trainable=trainable)
        self.k2 = self.add_weight(name='k', shape = (), initializer=initializer0, trainable=trainable)
    def call(self, inputs):
        return param_elliot_function(inputs, self.k1, self.k2 )


#%%
def hexr_self (x_unlab, p_m, alpha, parameters):
  """Self-supervised learning part in our framework.
  
  Args:
    x_unlab: unlabeled feature
    p_m: corruption probability
    alpha: hyper-parameter to control the weights of feature and mask losses
    parameters: epochs, batch_size
  Returns:
    encoder: Representation learning block
  """
    
  # Parameters
  _, dim = x_unlab.shape
  epochs = parameters['epochs']
  batch_size = parameters['batch_size']

  # Build model  
  inputs = Input(shape=(dim,))
  # Encoder  
  h = Dense(dim)(inputs)  # hidden layer

  """ Parameterized Elliot function"""
  Elliot = ParamElliotfn()
  h = Elliot(h)

  # Mask estimator - predict the mask (binary)
  output_1 = Dense(dim, activation='sigmoid', name = 'mask')(h)  

  # Feature estimator - using a linear activation function before using log cosh recon loss.
  output_2 = Dense(dim, activation='linear', name = 'feature')(h)
  
  model = Model(inputs = inputs, outputs = [output_1, output_2])
  
  model.compile(optimizer='rmsprop',loss={'mask': 'binary_crossentropy','feature': log_cosh},loss_weights={'mask':1, 'feature':alpha})
  
  # Generate corrupted samples
  m_unlab = mask_generator(p_m, x_unlab)
  m_label, x_tilde = pretext_generator(m_unlab, x_unlab)
  
  # Fit model on unlabeled data
  model.fit(x_tilde, {'mask': m_label, 'feature': x_unlab},epochs = epochs, batch_size= batch_size)
      
  # Extract encoder part
  layer_name = model.layers[1].name
  layer_output = model.get_layer(layer_name).output
  encoder = Model(inputs=model.input, outputs=layer_output)

  print("Proposed SLAB framework is trained.")
  
  return encoder