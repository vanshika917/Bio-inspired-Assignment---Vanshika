# -*- coding: utf-8 -*-
"""
Created on Wed Aug 1 18:15:09 2024

@author: vanshika gupta
"""

import numpy as np
import gymnasium as gym
import random
from datetime import datetime
import math
import csv
import matplotlib.pyplot as plt
from IPython.display import clear_output
from collections import namedtuple, deque
import os
import glob
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

""" 
Setup
"""
Device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using \"{Device}\" device.")

# Setting 'rgb_array' Render_mode to run the dqn network on image data
env = gym.make("FrozenLake-v1", render_mode='rgb_array')
# env = gym.make("FrozenLake-v1", render_mode='rgb_array', map_name="8x8")

env.reset()

"""
Hyper Parameters
"""
tau = 0.0005
gamma = 0.99
alpha = [1e-4]
batch_size = 128
epsilon_beg = 0.9
epsilon_end = 0.001
epsilon_decay = 1000
max_episodes = 10000
max_steps = 30

"""
Replay Memory
"""
Transition = namedtuple('Transition',('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        if (len(self.memory) < self.batch_size):
            return None
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)

"""
DQN Architecture
"""
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(9 * 9 * 64, 512),
            nn.LeakyReLU(True),
            nn.Linear(512, n_actions),
        )
        # self.model = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=8, stride=4),
        #     # nn.BatchNorm2d(16),
        #     # nn.MaxPool2d(kernel_size=4,stride=6),
        #     nn.ReLU(True),
        #     nn.Conv2d(16, 32, kernel_size=4, stride=2),
        #     # nn.BatchNorm2d(32),
        #     # nn.MaxPool2d(kernel_size=2,stride=4),
        #     nn.ReLU(True),
        #     # nn.Conv2d(32, 32, kernel_size=3, stride=1),
        #     # # nn.BatchNorm2d(32),
        #     # # nn.MaxPool2d(kernel_size=3,stride=1),
        #     # nn.ReLU(True),
        #     nn.Flatten(),
        #     nn.Linear(11*11*32, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, n_actions),
        # )

    def forward(self, x):
        return self.model(x)

"""
DQN Agent
"""
class RLAgent():
    def __init__(self, env, policy_net, target_net, optimizer,batch_size):
        self.env = env
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.batchsize = batch_size
        self.memory = ReplayMemory(capacity=10_000, batch_size=self.batchsize)
        self.tau = tau
        self.gamma = gamma
        self.checkpoint_freq = 1000
        self.reward_history = []
        self.loss_history = []
        self.steps_history = []
        self.epsilon_history = []
        self.max_episodes = 0
        self.epsilon_beg = epsilon_beg
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def is_closer_to_goal(self, state, next_state):
        """
        Determine if the agent has moved closer to the goal state.
        This is a simple example function. In a more complex environment,
        this function might consider distances or other metrics.
        """
        goal_position = 15  
        return abs(next_state - goal_position) < abs(state - goal_position)

    def train(self, n_episodes, n_steps, resume_training=True, pretrained_model=None,learning_rate=None,csv_file=None):
        """
        Train the Frozen Lake Agent

        n_episodes: Total episodes to train for
        n_steps: Total steps or actions before each episode is terminated
        save_dir: Directory where model is save at each checkpoint
        resume_training: Set to True to continue training from last checkpoint saved in save_dir

        return: None
        """
        self.max_episodes = n_episodes
        if resume_training:
            self.load_model(pretrained_model)

        self.q_value_history = {state_idx: [] for state_idx in range(env.observation_space.n)}
        for episode in range(n_episodes):
            # Save model every 20 episodes
            if (episode > 0 and episode % self.checkpoint_freq == 0):
                self.save_model(episode)

            state, info = self.env.reset()
            episode_reward = 0
            steps = 0
            total_loss = 0
            print(f"Start Episode: {episode}")

            eps_threshold = self.epsilon_end + (self.epsilon_beg - self.epsilon_end) * math.exp(-1 * episode / self.epsilon_decay)
            self.epsilon_history.append(eps_threshold)  # Track epsilon

            for step in range(n_steps):
                steps += 1  # Increment step count

                state_img = extract_state_img(self.env, state, transforms=transforms).to(Device)

                # Select an action via explore vs exploit
                sample = random.random()
                # eps_threshold = epsilon_end + (epsilon_beg - epsilon_end) * math.exp(-1 * episode / epsilon_decay)
                if sample > eps_threshold or resume_training:
                    with torch.no_grad():
                        action = torch.argmax(self.policy_net(state_img.unsqueeze(dim=0))).item()
                else:
                    action = self.env.action_space.sample()

                # Execute action, observe reward, and store experience
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                if state == 0:
                    with torch.no_grad():
                        q_values = self.policy_net(state_img.unsqueeze(dim=0)).cpu().numpy().flatten()
                    self.q_value_history[state].append(q_values)
                done = terminated or truncated
                if terminated:
                    next_state = None

                title = f"Episode:{episode}    Step: {step}"
                render_env(self.env, title)
                self.memory.push(state, action, reward, next_state)

                state = next_state

                # Optimize the model and accumulate the loss
                loss = self.optimize_model()
                if loss is not None:
                    total_loss += loss.item()

                # Soft update target network's weights
                policy_net_dict = self.policy_net.state_dict()
                target_net_dict = self.target_net.state_dict()
                for key in policy_net_dict:
                    target_net_dict[key] = policy_net_dict[key] * self.tau + target_net_dict[key] * (1 - self.tau)
                self.target_net.load_state_dict(target_net_dict)

                # Done
                if done:
                    break
            self.average_loss = total_loss / steps
            self.loss_history.append(self.average_loss)
            self.reward_history.append(episode_reward)
            self.steps_history.append(steps)

            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                # loss_val = loss.item()
                writer.writerow([learning_rate,episode + 1, episode_reward,np.cumsum(self.reward_history)[-1], self.average_loss, eps_threshold, steps])

            self.plot_training(episode)

        
    def test(self, n_episodes=100, n_steps=10, pretrained_model=None,learning_rate=None):
        """
        Test

        n_episodes: Total episodes to test for
        n_steps: Total steps or actions before each episode is terminated
        model_dir: Directory where model is saved

        return: None
        """
        if pretrained_model:
            self.load_model(pretrained_model)
        
        n_success = 0
        n_failures = 0
        
        for episode in range(n_episodes):
            state, info = self.env.reset()
            episode_reward = 0 
            steps = 0
            
            for step in range(n_steps):
                state_img = extract_state_img(self.env, state, transforms=transforms).to(Device)
                action = torch.argmax(self.policy_net(state_img.unsqueeze(dim=0))).item()
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                steps += 1
                
                if reward > 0:
                    n_success += 1
                elif reward == 0 and terminated:
                    n_failures += 1
                
                done = terminated or truncated
                if terminated:
                    next_state = None

                render_env(self.env, f"Episode: {episode}, Step: {step}")
                state = next_state
                
                if done:
                    break

            # Log episode data
            self.episode_data.append({
                'alpha':learning_rate,
                'Episode': episode,
                'Total Reward': episode_reward,
                'Total Steps': steps,
                'Success': n_success,
                'Failures': n_failures,
            })
        print(f"Accuracy: {n_success/n_episodes}")
        print(f"Failures: {n_failures/n_episodes}")
        print(f"Truncations: {(n_episodes - n_success - n_failures)/n_episodes}")
    
        
    def optimize_model(self):
        """
        Optimize the model based on loss between policy net and target net

        """
        # Sample random batch of states
        transitions = self.memory.sample()
        if transitions is None:
            return None

        batch = Transition(*zip(*transitions))
        reward_batch = torch.tensor([reward for reward in batch.reward]).to(Device)
        action_batch = torch.tensor([action for action in batch.action]).to(Device)
        action_batch = torch.reshape(action_batch, (self.memory.batch_size, 1))

        # Get Q values predicted by the policy net
        current_state_imgs = torch.zeros(size=(len(batch.state), 1, image_width, image_width), dtype=torch.float).to(Device)
        for i, state in enumerate(batch.state):
            current_state_imgs[i] = extract_state_img(self.env, state, transforms=transforms)
        predicted_q_values = self.policy_net(current_state_imgs).gather(1, action_batch)

        # Get Q' values as predicted by the target net
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=Device, dtype=torch.bool)
        non_final_next_states = torch.tensor([s for s in batch.next_state if s is not None]).to(Device)
        next_state_imgs = torch.zeros(size=(len(non_final_next_states), 1, image_width, image_width), dtype=torch.float).to(Device)
        for i, state in enumerate(non_final_next_states):
            next_state_imgs[i] = extract_state_img(self.env, state, transforms=transforms)

        expected_q_values = torch.zeros(size=(len(batch.state),)).to(Device)
        with torch.no_grad():
            expected_q_values[non_final_mask] = self.target_net(next_state_imgs).max(1)[0]
            # next_state_actions = self.policy_net(next_state_imgs).max(1)[1]
            # expected_q_values[non_final_mask] = self.target_net(next_state_imgs).gather(1, next_state_actions.unsqueeze(1)).squeeze()

        expected_q_values = (expected_q_values * self.gamma) + reward_batch
        expected_q_values = expected_q_values.unsqueeze(1)

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(predicted_q_values, expected_q_values)
        # print(loss.item())
        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss

    def save_model(self, checkpoint_num):
        """
        Saves a model, and defines model name as checkpoint number
        """
        # Set up path
        timestamp = int(datetime.now().timestamp())
        filename = "model_" + str(checkpoint_num) + "_" + str(timestamp) + ".pt"
        directory = 'checkpts_int_rewards'
        path = os.path.join(directory, filename)

        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save state dicts for policy and target net
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
        }, path)


    def load_model(self, filename):
        """
        Load a model

        dir: Directory where the model is saved

        :return: None
        """

        # Load model
        state_dicts = torch.load(filename, map_location=Device)
        self.policy_net.load_state_dict(state_dicts['policy_net_state_dict'])
        self.target_net.load_state_dict(state_dicts['target_net_state_dict'])
        self.optimizer.load_state_dict(state_dicts['optim_state_dict'])

        # Put models in eval mode
        self.policy_net.eval()
        self.target_net.eval()

    def plot_training(self, episode, window_size=50):
        """
        Plots the training rewards, losses, steps, and epsilon decay with an optional window size for the SMA.

        episode: The current episode number.
        window_size: The number of episodes over which to calculate the moving average.
        """

        # Rewards
        if len(self.reward_history) >= window_size:
            sma_rewards = np.convolve(self.reward_history, np.ones(window_size)/window_size, mode='valid')
        else:
            sma_rewards = self.reward_history  # Not enough data for SMA

        plt.figure()
        plt.title("Rewards")
        plt.plot(self.reward_history, label='Raw Reward', color='#F6CE3B', alpha=1)
        if len(sma_rewards) > 1:
            plt.plot(sma_rewards, label=f'SMA {window_size}', color='#385DAA')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()

        if episode == self.max_episodes:
            plt.savefig('./reward_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close()

        # Cumulative Rewards
        cumulative_rewards = np.cumsum(self.reward_history)

        if len(cumulative_rewards) >= window_size:
            sma_cumulative_rewards = np.convolve(cumulative_rewards, np.ones(window_size)/window_size, mode='valid')
        else:
            sma_cumulative_rewards = cumulative_rewards  # Not enough data for SMA

        plt.figure()
        plt.title("Cumulative Rewards")
        plt.plot(cumulative_rewards, label='Cumulative Reward', color='#F6CE3B', alpha=1)
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Rewards")
        plt.legend()

        if episode == self.max_episodes:
            plt.savefig('./cumulative_reward_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close()
        
        # Loss
        if len(self.loss_history) >= window_size:
            sma_losses = np.convolve(self.loss_history, np.ones(window_size)/window_size, mode='valid')
        else:
            sma_losses = self.loss_history

        plt.figure()
        plt.title("Loss")
        plt.plot(self.loss_history, label='Raw Loss', color='#CB291A', alpha=1)
        if len(sma_losses) > 1:
            plt.plot(sma_losses, label=f'SMA {window_size}', color='#85C1E9')
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.legend()

        if episode == self.max_episodes:
            plt.savefig('./loss_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close()

        # Plot Epsilon over time
        plt.figure()
        plt.title("Epsilon Decay")
        plt.plot(self.epsilon_history, label='Epsilon', color='#FF5733', alpha=1)
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        if episode == self.max_episodes:
            plt.savefig('./epsilon_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.close()
        
        # Steps
        plt.figure()
        plt.title("Steps per Episode")
        plt.plot(self.steps_history, label=f'Steps', color='#1F618D')
        plt.xlabel("Episode")
        plt.ylabel("Average Steps")
        
        if episode == self.max_episodes:
            plt.savefig('./steps_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close()

"""
Image Processing
"""
# Use a global plot to support live rendering
figure, ax = plt.subplots()
image = ax.imshow(env.render())
plt.show(block=False)

def render_env(env, title=None):
    """
    Render environment image.
    env: Frozenlake environment
    """

    if title:
        figure.suptitle(title)

    image.set_data(env.render())
    figure.canvas.draw()
    plt.pause(0.0001)

def render_state(env, state_idx, transforms):
    """
    Render state image.
    env: Frozenlake environment
    state_idx: Index of the state, range is 0-15
    transforms: Transform the image before return
    """
    image = extract_state_img(env, state_idx, transforms).permute(1,2,0)
    plt.imshow(image, cmap='gray')

def extract_state_img(env, state_idx, transforms):
    """
    Extracts the state image from the environment image.

    env: Frozenlake environment
    state_idx: Index of the state, range is 0-15
    transforms: Transform the image before return

    return: Image of shape CxHxW
    """
    # Convert env rgb array to tensor
    env = torch.tensor(env.render())
    block_size = env.shape[0] // 4
    # block_size = env.shape[0] // 8
    # Extract state from given index
    env = env.permute(2, 0, 1)
    env = transforms(env)
    env = env.permute(1, 2, 0)

    start_idx = (state_idx // 4) * block_size
    end_idx = (state_idx % 4) * block_size
    # start_idx = (state_idx // 8) * block_size
    # end_idx = (state_idx % 8) * block_size

    state_img = env[start_idx:(start_idx + block_size + 2 * padding), end_idx:(end_idx + block_size + 2 * padding), :]

    state_img = state_img.permute(2, 0, 1).type(torch.float)
    return state_img

"""
Run
"""

padding = 20

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Pad(padding=padding, fill=225),
    torchvision.transforms.Grayscale(1),
    torchvision.transforms.Lambda(lambda x: x/255.0),
])

image_width = extract_state_img(env, state_idx=9, transforms=transforms).shape[1]

start_time = time.time()

# DQN Networks
for i in range(len(alpha)):
    csv_file = f'training_.csv'

    # Write the header for the CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['LR', 'episode', 'episode_reward','cumulative reward', 'loss', 'eps_threshold', 'steps'])

    policy_net = DQN(image_width * image_width, env.action_space.n).to(Device)
    target_net = DQN(image_width * image_width, env.action_space.n).to(Device)
    # optimizer = torch.optim.AdamW(policy_net.parameters(), lr=alpha[i], amsgrad=True)
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=alpha[i])

    target_net.load_state_dict(policy_net.state_dict())

    # Initialize
    env.reset()
    dqn_trainer = RLAgent(env, policy_net, target_net, optimizer, batch_size=batch_size)

    # Train
    # dqn_trainer.train(3000, 20, resume_training=True, pretrained_model='checkpts_int_rewards\model_900_1723938252.pt')
    dqn_trainer.train(10000, 30, resume_training=False, learning_rate=alpha[i], csv_file=csv_file)

# Test
dqn_trainer.test(n_episodes=5000, n_steps=30)

elapsed_time = time.time() - start_time

hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Elapsed Time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")