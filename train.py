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
REPLAY_MEMORY_SIZE = 500
EPISODE = 3000
STEP = 200
BATCH_SIZE = 50
GAMMA = 0.995
TEST_FREQUENCY = 100
TEST = 3
epsilon = 1.0
epsilon_discount_factor = 0.995

print(f'STATE_DIM = {STATE_DIM}, ACTION_DIM = {ACTION_DIM}')

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
        action = np.random.random_integers(0, ACTION_DIM - 1)
    print(f'Recommending action {action}')
    return action

# Function to determine the next action to take (without epsilon-greedy).
def get_action_deterministic(state):
    action = np.argmax(model.predict(state))
    return action

initial_state = env.reset().reshape((1, STATE_DIM))

replay_buffer = deque(maxlen=REPLAY_MEMORY_SIZE)

for episode in range(EPISODE):
    print(f'Episode {episode} of {EPISODE}')
    state = env.reset()
    step = 0
    for step in range(STEP):
        action = get_action(state.reshape((1, STATE_DIM)), epsilon)
        next_state, reward, is_done, _ = env.step(action)
        if len(replay_buffer) < REPLAY_MEMORY_SIZE:
            replay_buffer.append((state, action, reward, next_state, is_done))
        if is_done:
            break
        state = next_state
    # After sampling another experience, we check if we have enough experience to perform training.
    if len(replay_buffer) >= BATCH_SIZE:
        sample_replays = random.sample(replay_buffer, BATCH_SIZE)
        for replay in sample_replays:
            (train_state, train_action, train_reward, train_next_state, train_is_done) = replay
            train_next_state = train_next_state.reshape((1, STATE_DIM))
            estimated_q_values = model.predict(train_next_state)
            target_q_value = train_reward
            if train_is_done:
                target_q_value += GAMMA * np.max(estimated_q_values)
            # Train the network on this sample.
            model.fit([train_state.reshape((1, STATE_DIM))], [estimated_q_values], 1)
    epsilon *= epsilon_discount_factor
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = get_action_deterministic(state.reshape((1, STATE_DIM)))
                state, reward, done, _ = env.step(action)
                if done:
                    break
