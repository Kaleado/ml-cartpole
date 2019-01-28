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
model = Sequential()
model.add(Dense(64, activation='tanh', input_shape=(STATE_DIM,)))
model.add(Dense(ACTION_DIM, activation='sigmoid'))

# Define our loss and optimiser functions.
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# We get the Q-value of our desired action.

# Function to determine the next action to take.
def get_action(state, epsilon):
    if(np.random.random() > epsilon):
        action = np.argmax(model.predict(state))
    else:
        action = np.random.random_integers(0, ACTION_DIM)
    return action

initial_state = env.reset().reshape((1, STATE_DIM))
print(STATE_DIM, initial_state.shape)
print(get_action(initial_state, 0.0))
