import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from typing import Callable
from tqdm.notebook import tqdm

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_datasets as tfds 
import torch

import jax
import optax
import flax
import jax.numpy as jnp
from jax import jit
from jax import lax
from jax_resnet import pretrained_resnet, slice_variables, Sequential
from flax.jax_utils import replicate, unreplicate
from flax.training import train_state, checkpoints
from flax import linen as nn
from flax.core import FrozenDict,frozen_dict
from flax.training.common_utils import shard

from functools import partial

# TODO: Remove Tensorflow references

############## CONFIG FOR TRAINING ############## 

Config = {
    'NUM_LABELS': 10,
    'N_SPLITS': 5,
    'BATCH_SIZE': 32,
    'N_EPOCHS': 10,
    'LR': 0.001,
    'WIDTH': 32,
    'HEIGHT': 32,
    'IMAGE_SIZE': 128,
    'WEIGHT_DECAY': 1e-5,
    'FREEZE_BACKBONE': True
}

#####################################################


##################### DEFINING CUSTOM MODEL #####################
"""
reference - https://www.kaggle.com/code/alexlwh/happywhale-flax-jax-tpu-gpu-resnet-baseline
"""

class Head(nn.Module):
    '''head model'''
    batch_norm_cls: partial = partial(nn.BatchNorm, momentum=0.9)
    @nn.compact
    def __call__(self, inputs, train: bool):
        output_n = inputs.shape[-1]
        x = self.batch_norm_cls(use_running_average=not train)(inputs)
        x = nn.Dropout(rate=0.25)(x, deterministic=not train)
        x = nn.Dense(features=output_n)(x)
        x = nn.relu(x)
        x = self.batch_norm_cls(use_running_average=not train)(x)
        x = nn.Dropout(rate=0.5)(x, deterministic=not train)
        x = nn.Dense(features=Config["NUM_LABELS"])(x)
        return x

class Model(nn.Module):
    '''Combines backbone and head model'''
    backbone: Sequential
    head: Head
        
    def __call__(self, inputs, train: bool):
        x = self.backbone(inputs)
        # average pool layer
        x = jnp.mean(x, axis=(1, 2))
        x = self.head(x, train)
        return x


class TrainState(train_state.TrainState):
    batch_stats: FrozenDict
    loss_fn: Callable = flax.struct.field(pytree_node=False)
    eval_fn: Callable = flax.struct.field(pytree_node=False)


############################################################



############## PREPROCESSING AND LOADING DATA ############## 

# Transform the input images
def transform_images(row, size):
    '''
    Resize image 
    INPUT row , size
    RETURNS resized image and its label
    '''
    x_train = tf.image.resize(row['image'], (size, size))
    return x_train, row['label']

# Load in the trianing and test datasets
def load_datasets():
    '''
    load and transform dataset from tfds
    RETURNS train and test dataset
    
    '''
    
    # Construct a tf.data.Dataset
    train_ds,test_ds = tfds.load('cifar10', split=['train','test'], shuffle_files=True)

    train_ds = train_ds.map(lambda row:transform_images(row,Config["IMAGE_SIZE"]))
    test_ds = test_ds.map(lambda row:transform_images(row,Config["IMAGE_SIZE"]))
    
    # Build your input pipeline
    train_dataset = train_ds.batch(Config["BATCH_SIZE"]).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_ds.batch(Config["BATCH_SIZE"]).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset,test_dataset

#################################################################



##################### LOADING PRETRAINED MODEL ##################

def _get_backbone_and_params(model_arch: str):
    '''
    Get backbone and params
    1. Loads pretrained model (resnet18)
    2. Get model and param structure except last 2 layers
    3. Extract the corresponding subset of the variables dict
    INPUT : model_arch
    RETURNS backbone , backbone_params
    '''
    if model_arch == 'resnet18':
        resnet_tmpl, params = pretrained_resnet(18)
        model = resnet_tmpl()
    else:
        raise NotImplementedError
        
    # get model & param structure for backbone
    start, end = 0, len(model.layers) - 2
    backbone = Sequential(model.layers[start:end])
    backbone_params = slice_variables(params, start, end)
    return backbone, backbone_params


def get_model_and_variables(model_arch: str, head_init_key: int):
    '''
    Get model and variables 
    1. Initialise inputs(shape=(1,image_size,image_size,3))
    2. Get backbone and params
    3. Apply backbone model and get outputs
    4. Initialise head
    5. Create final model using backbone and head
    6. Combine params from backbone and head
    
    INPUT model_arch, head_init_key
    RETURNS  model, variables 
    '''
    
    #backbone
    inputs = jnp.ones((1, Config['IMAGE_SIZE'],Config['IMAGE_SIZE'], 3), jnp.float32)
    backbone, backbone_params = _get_backbone_and_params(model_arch)
    key = jax.random.PRNGKey(head_init_key)
    backbone_output = backbone.apply(backbone_params, inputs, mutable=False)
    
    #head
    head_inputs = jnp.ones((1, backbone_output.shape[-1]), jnp.float32)
    head = Head()
    head_params = head.init(key, head_inputs, train=False)
    
    #final model
    model = Model(backbone, head)
    variables = FrozenDict({
        'params': {
            'backbone': backbone_params['params'],
            'head': head_params['params']
        },
        'batch_stats': {
            'backbone': backbone_params['batch_stats'],
            'head': head_params['batch_stats']
        }
    })
    return model, variables
  
####################################################################



##################### TRAINING HELPER FUNCTIONS #####################

"""
reference - https://github.com/deepmind/optax/issues/159#issuecomment-896459491
"""
def zero_grads():
    '''
    Zero out the previous gradient computation
    '''
    def init_fn(_): 
        return ()
    def update_fn(updates, state, params=None):
        return jax.tree_map(jnp.zeros_like, updates), ()
    return optax.GradientTransformation(init_fn, update_fn)

"""
reference - https://colab.research.google.com/drive/1g_pt2Rc3bv6H6qchvGHD-BpgF-Pt4vrC#scrollTo=TqDvTL_tIQCH&line=2&uniqifier=1
"""
def create_mask(params, label_fn):
    def _map(params, mask, label_fn):
        for k in params:
            if label_fn(k):
                mask[k] = 'zero'
            else:
                if isinstance(params[k], FrozenDict):
                    mask[k] = {}
                    _map(params[k], mask[k], label_fn)
                else:
                    mask[k] = 'optim'
    mask = {}
    _map(params, mask, label_fn)
    return frozen_dict.freeze(mask)

def create_train_state(model, variables, optimizer, loss_fn, eval_fn):
  return TrainState.create(
      apply_fn = model.apply,
      params = variables['params'],
      tx = optimizer,
      batch_stats = variables['batch_stats'],
      loss_fn = loss_fn,
      eval_fn = eval_fn
  )

def accuracy(logits,labels):
    '''
    calculates accuracy based on logits and labels
    INPUT logits , labels
    RETURNS accuracy
    '''
    return [jnp.mean(jnp.argmax(logits, -1) == labels)]

####################################################################



##################### TRAIN/VAL/TEST STEP FUNCTIONS ################


def train_step(state: TrainState, batch, labels, dropout_rng):
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
    
    # params as input because we differentiate wrt it 
    def loss_function(params):
        # if you set state.params, then params can't be backpropagated through!
        variables = {'params': params, 'batch_stats': state.batch_stats}
        
        # return mutated states if mutable is specified
        logits, new_batch_stats = state.apply_fn(
            variables, batch, train=True, 
            mutable=['batch_stats'],
            rngs={'dropout': dropout_rng}
        )
        # logits: (BS, OUTPUT_N), one_hot: (BS, OUTPUT_N)
        one_hot = jax.nn.one_hot(labels,Config["NUM_LABELS"])
        loss = state.loss_fn(logits, one_hot).mean()
        return loss, (logits, new_batch_stats)
    
    
    # backpropagation and update params & batch_stats 
    grad_fn = jax.value_and_grad(loss_function, has_aux=True) #differentiate the loss function
    (loss, aux), grads = grad_fn(state.params)
    logits, new_batch_stats = aux
    grads = lax.pmean(grads, axis_name='batch') #compute the mean gradient over all devices
    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_batch_stats['batch_stats'] #applies the gradients to the weights.
    )
    
    # evaluation metrics
    accuracy = state.eval_fn(logits, labels)
    
    # store metadata
    metadata = jax.lax.pmean(
        {'loss': loss, 'accuracy': accuracy},
        axis_name='batch'
    )
    return new_state, metadata, new_dropout_rng


def val_step(state: TrainState, batch, labels):
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits = state.apply_fn(variables, batch, train=False) # stack the model's forward pass with the logits function
    return state.eval_fn(logits, labels)

def test_step(state: TrainState, batch):
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits = state.apply_fn(variables, batch, train=False) # stack the model's forward pass with the logits function
    return logits

####################################################################