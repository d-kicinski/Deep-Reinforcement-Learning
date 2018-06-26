import random
import collections
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np



class ReplayBuffer(object):
    """Dataset of transitions."""
    Transition = collections.namedtuple(
        'Transition', ('state', 'action', 'state_next', 'reward'))

    def __init__(self, capacity=10000):
        self._capacity = capacity
        self._buffer = collections.deque()

    def push(self, *args):
        """Saves a transition, evict old data."""
        if len(self._buffer) >= self._capacity:
            self._buffer.popleft()
        self._buffer.append(self.Transition(*args))

    def sample_batch(self, batch_size):
        return random.sample(self._buffer, batch_size)

    def __len__(self):
        return len(self._buffer)



class QAproximator(nn.Module):
    def __init__(self,
                 D_in,
                 D_out,
                 H,
                 hidden_layers,
                 batchnorm=False,
                 linear=False):

        nn.Module.__init__(self)

        self.linear = linear
        self.batchnorm = batchnorm

        #self.layers = []
        self.layers = nn.ModuleList()

        self.batchnorm_layers = []

        # Add the very first layer
        self.layers.append(nn.Linear(D_in, H))  # bottom
        if batchnorm:
            self.batchnorm_layers.append(nn.BatchNorm1d(H))

        # Add middle hidden_layers
        for l in range(hidden_layers - 1):
            self.layers.append(nn.Linear(H, H))
            if batchnorm:
                self.batchnorm_layers.append(nn.BatchNorm1d(H))

        # The very end - no nonlinearity  at the end
        self.head = nn.Linear(H, D_out)

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
            if self.batchnorm:
                x = self.batchnorm_layers[i](x)
            if not self.linear:
                x = F.relu(x)
        return self.head(x)



class Optimizer(object):
    def __init__(self, policy_net, target_net, replay_buffer, params_dqn, params_optim):
        self.policy_net = policy_net
        self.target_net = target_net

        self.replay_buffer = replay_buffer

        self.p_dqn = params_dqn
        self.p_optim = params_optim

        # Optimizer for lazy people
        self.optimizer = optim.Adam(policy_net.parameters(), self.p_optim.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.p_optim.step_lr, gamma=0.5)

    def step(self, timesteps):

        if len(self.replay_buffer) < self.p_optim.batch_size:
            return


        # NOTE: MOVE THIS CODE TO BATCH SAMPLING IN REPLAY BUFFER
        transitions = self.replay_buffer.sample_batch(self.p_optim.batch_size)
        batch_state, batch_action, batch_state_next, batch_reward = zip(*transitions)

        # Convert tuple of tensors into one tensor, treat each tuple as row
        batch_state = torch.cat(batch_state)
        batch_action = torch.cat(batch_action)
        batch_reward = torch.cat(batch_reward)
        batch_state_next = torch.cat(batch_state_next)



        q_values = self.policy_net(batch_state).gather(1, batch_action)

        if self.p_dqn.use_double_dqn:

            actions = self.policy_net(batch_state_next).detach().max(1)[1]
            # expected Q values are estimated from actions which gives maximum
            # value by current policy
            max_next_q_values = self.target_net(batch_state_next).detach().gather(1, actions.unsqueeze(1)).squeeze()

            expected_q_values = batch_reward + (self.p_dqn.decay_rate * max_next_q_values)
        else:
            #  values are estimated from actions which gives maximum value by target policy
            max_next_q_values = self.target_net(batch_state_next).detach().max(1)[0]
            expected_q_values = batch_reward + (self.p_dqn.decay_rate * max_next_q_values)

        # Huber loss but weird name
        loss = F.smooth_l1_loss(q_values.view(-1), expected_q_values.view(-1))

        # Finally perform one gradient step
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient will be high due to the bellman error loss, so clamp it
        # NOTE 1:btw i think Huber loss might be enough so try without it
        # NOTE 2: doesn't work btw
        if self.p_optim.use_gradient_clipping:
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)

        self.optimizer.step()


        # Update target parameters by choosen method
        if self.p_dqn.use_polyak_averaging:
            beta = self.p_dqn.polyak_interpolation_rate
            params1 = self.target_net.named_parameters()
            params2 = self.policy_net.named_parameters()

            dict_params2 = dict(params2)

            for name1, param1 in params1:
                if name1 in dict_params2:
                    dict_params2[name1].data.copy_(beta*param1.data + (1-beta)*dict_params2[name1].data)
            self.target_net.load_state_dict(dict_params2)

        elif timesteps % self.p_dqn.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss


class ActionChooser(object):
    def __init__(self, policy, start, end, decay ):
        self.policy = policy
        self.start = start
        self.end = end
        self.decay = decay

    def normal_policy(self, state, timestep):
        return self.policy(state).max(1)[1].view(1, 1)

    def epsilon_greedy_policy(self, state, timestep):

        sample = random.random()
        threshold = self.end + (
            self.start - self.end) * math.exp(-1. * timestep / self.decay)

        if sample > threshold:
            with torch.no_grad():
                return self.policy(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(2)]], dtype=torch.long)
