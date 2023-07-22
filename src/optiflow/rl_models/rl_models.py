# Generate DEEPQ Model to optimize inventory replenishment decisions using pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple
from itertools import count
from torch.autograd import Variable
from optiflow.utils import *
from optiflow.rl_models import *
# import types tensors from torch
from torch import FloatTensor, LongTensor, ByteTensor

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

# Generate trainer for DQN model
class DQNTrainer():
    def __init__(self, env, model, optimizer, criterion, num_episodes, batch_size, gamma, epsilon, epsilon_decay, epsilon_min, target_update, memory_size, render=False):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update = target_update
        self.memory_size = memory_size
        self.render = render
        self.memory = ReplayMemory(memory_size)
        self.steps_done = 0
        self.episode_durations = []
        self.episode_rewards = []

    def select_action(self, state):
        sample = random.random()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if sample > self.epsilon:
            return self.model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
        else:
            return LongTensor([[random.randrange(self.env.action_space.n)]])

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True).type(FloatTensor)
        state_batch = Variable(torch.cat(batch.state)).type(FloatTensor)
        action_batch = Variable(torch.cat(batch.action)).type(LongTensor)
        reward_batch = Variable(torch.cat(batch.reward)).type(FloatTensor)
        state_action_values = self.model(state_batch).gather(1, action_batch)
        next_state_values = Variable(torch.zeros(self.batch_size).type(FloatTensor))
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = self.criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)


        # What is ReplayMemory? 
        # ReplayMemory is a class that stores the transitions that the agent observes, allowing us to reuse this data later.
        # By sampling from it randomly, the transitions that build up a batch are decorrelated.
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)
    

class Transition(object):
    def __init__(self, state, action, next_state, reward):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

#