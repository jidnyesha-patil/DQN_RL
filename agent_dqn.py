#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import tensor

from agent import Agent
from dqn_model import DQN

from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')
        
        # general parameters
        self.is_cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.is_cuda_available else "cpu")
        self.step = 0
        self.run_name = 'dqn_model'
        self.model_save_path = 'models'
        self.model_save_interval = 500
        self.log_path = 'train_log.out'
        self.tensorboard_summary_path = 'tensorboard_summary'
        
        # Environment and network parameters
        self.env = env
        self.num_actions = env.action_space.n
        self.action_list = np.arange(self.num_actions)
        self.metrics_capture_window = args.metrics_capture_window
        self.replay_buffer_size = args.replay_buffer_size
        self.replay_memory = deque([], self.replay_buffer_size)
        self.position = 0
        self.total_num_steps = args.total_num_steps
        self.episodes = args.episodes
        self.gamma = args.gamma
        self.learning_rate = args.learning_rate
        self.initial_epsilon = args.initial_epsilon
        self.final_epsilon = args.final_epsilon
        self.epsilon = self.initial_epsilon
        self.steps_to_explore = args.steps_to_explore
        self.epsilon_step = (self.initial_epsilon - self.final_epsilon) / self.steps_to_explore
        self.network_update_interval = args.network_update_interval
        self.network_train_interval = args.network_train_interval
        self.last_n_rewards = deque([], self.metrics_capture_window)
        self.start_to_learn = args.start_to_learn
        self.ddqn = args.ddqn

        self.batch_size = args.batch_size
        self.q_network = DQN().to(self.device)
        self.target_q_network = DQN().to(self.device)
        self.loss_function = F.smooth_l1_loss
        self.optimiser = optim.Adam(self.q_network.parameters(), lr = args.learning_rate)
        self.probability_list = np.zeros(env.action_space.n, np.float32)
        self.q_network.train()
        self.target_q_network.eval()
        self.mode = "Random"
        self.state_counter_while_testing = 0
        self.model_test_path=args.model_test_path

        # create necessary paths
        self.create_dirs()
        # self.q_network.load_state_dict(torch.load("dqn_model_34500.pt", map_location=self.device))
        # test mode
        if args.test_dqn:
            print('loading trained model')
            self.q_network.load_state_dict(torch.load(self.model_test_path, map_location=self.device))

        self.log_file = open(self.model_save_path + '/' + self.run_name + '.log', 'w') if not args.test_dqn else None

        # Set target_q_network weight
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.writer = SummaryWriter(args.tensorboard_summary_path)
            

    def create_dirs(self):
        paths = [self.model_save_path, self.tensorboard_summary_path]
        [os.makedirs(path) for path in paths if not os.path.exists(path)]
        
        
    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        self.state_counter_while_testing += 1
    
    
    def make_action(self, observation, state_count=0, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        self.init_game_setting()
        with torch.no_grad():
            if test:
                observation = torch.reshape(tensor(observation, dtype=torch.float32), [1, 84, 84, 4]).permute(0, 3, 1, 2).to(self.device)
                if state_count < 5000:
                    action = torch.argmax(self.q_network(tensor(observation).float()).detach())
                    return action.item()
                else:
                    return np.random.choice(range(self.num_actions))

            # Fill up probability list equal for all actions
            self.probability_list.fill(self.epsilon / self.num_actions)
            
            # Fetch q from the model prediction
            q, argq = self.q_network(tensor(observation).float()).data.cpu().max(1)
            
            # Increase the probability for the selected best action
            self.probability_list[argq[0].item()] += 1 - self.epsilon
            
            # Use random choice to decide between a random action / best action
            action = torch.tensor([np.random.choice(self.action_list, p=self.probability_list)])
            
        return action.item(), q

    
    def push(self, transition_tuple):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        self.replay_memory.appendleft(transition_tuple)
        
        
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, self.batch_size)
        return minibatch
        
        
    def optimize_network(self):
        # Sample random minibatch of transition from replay memory
        minibatch = self.replay_buffer()
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = map(lambda x: Variable(torch.cat(x, 0)), zip(*minibatch))
        state_batch, reward_batch, next_state_batch = state_batch.float(), reward_batch.float(), next_state_batch.float()
        
        q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        target_values = self.target_q_network(next_state_batch)
        
        if self.ddqn:
            best_actions = torch.argmax(self.q_network(next_state_batch), dim=-1)
            target_values = target_values.gather(1, tensor(best_actions).unsqueeze(1)).squeeze(1)
        else:
            target_values = target_values.max(1)[0].squeeze(0)

        target_values = target_values * self.gamma * (1 - terminal_batch)

        loss = self.loss_function(q_values, reward_batch + target_values)
        self.optimiser.zero_grad()
        loss.backward()
        for param in self.q_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimiser.step()
        return loss.cpu().detach().numpy()
        

    def log_summary(self, global_step, episode_loss, episode_reward):
        self.writer.add_scalar('Train/Episode Reward', sum(episode_reward), global_step)
        self.writer.add_scalar('Train/Average Loss', np.mean(episode_loss), global_step)
        self.writer.add_scalar('Train/Average reward(100)', np.mean(self.last_n_rewards), global_step)
        self.writer.flush()
        
        
    def train(self):
        """
        Implement your training algorithm here
        """
        for episode in range(self.episodes):
            state = self.env.reset()
            state = torch.reshape(tensor(state, dtype=torch.float32), [1, 84, 84, 4]).permute(0, 3, 1, 2).to(self.device)
            done = False
            episode_reward = []
            episode_loss = []

            # save network
            if episode % self.model_save_interval == 0:
                save_path = self.model_save_path + '/' + self.run_name + '_' + str(episode) + '.pt'
                torch.save(self.q_network.state_dict(), save_path)
                print('Successfully saved: ' + save_path)

            while not done:

                # update target network
                if self.step % self.network_update_interval == 0:
                    print('Updating target network')
                    self.target_q_network.load_state_dict(self.q_network.state_dict())

                if self.step > len(self.replay_memory):
                    self.epsilon = max(self.final_epsilon, self.initial_epsilon - self.epsilon_step * self.step)
                    if self.epsilon > self.final_epsilon:
                        self.mode = 'Explore'
                    else:
                        self.mode = 'Exploit'

                action, q = self.make_action(state, 0, test = False)
                next_state, reward, done, _ = self.env.step(action)

                next_state = torch.reshape(tensor(next_state, dtype=torch.float32), [1, 84, 84, 4]).permute(0, 3, 1, 2).to(self.device)
                
                self.push((state, torch.tensor([int(action)]), torch.tensor([reward], device=self.device), next_state, torch.tensor([done], dtype=torch.float32)))
                
                episode_reward.append(reward)
                
                self.step += 1
                
                state = next_state

                # train network
                if self.step >= self.start_to_learn and self.step % self.network_train_interval == 0:
                    loss = self.optimize_network()
                    episode_loss.append(loss)

                if done:
                    print('Episode:', episode, ' | Steps:', self.step, ' | Eps: ', self.epsilon, ' | Reward: ',
                          sum(episode_reward),
                          ' | Avg Reward: ', np.mean(self.last_n_rewards), ' | Loss: ',
                          np.mean(episode_loss), ' | Mode: ', self.mode)
                    print('Episode:', episode, ' | Steps:', self.step, ' | Eps: ', self.epsilon, ' | Reward: ',
                          sum(episode_reward),
                          ' | Avg Reward: ', np.mean(self.last_n_rewards), ' | Loss: ',
                          np.mean(episode_loss), ' | Mode: ', self.mode, file=self.log_file)
                    self.log_summary(episode, episode_loss, episode_reward)
                    self.last_n_rewards.append(sum(episode_reward))
                    episode_reward.clear()
                    episode_loss.clear()
