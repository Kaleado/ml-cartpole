import gym
import tensorflow as tf
import numpy as np
import math
import random
import keras
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Reshape, Dropout, Activation
from keras.optimizers import SGD
from collections import deque

# Set up OpenAI gym environment
ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# We create a two-layer network, consisting only of fully-connected layers.
# These will be predicting our Q-values.
state_input = tf.placeholder("float", [None, STATE_DIM])
model = Sequential()
model.add(InputLayer(input_tensor=state_input))
model.add(Dense(32, activation='tanh'))
model.add(Dense(ACTION_DIM, activation='sigmoid'))

# Define our loss and optimiser functions.
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# We get the Q-value of our desired action.
