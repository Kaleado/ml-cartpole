import gym
import tensorflow as tf
import numpy as np
import math
import random
from collections import deque

# Set up OpenAI gym environment
ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)

