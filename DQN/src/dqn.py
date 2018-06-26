import math
import random
import time
import os
import inspect
import datetime
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import gym
from gym import wrappers

import dqn_utils as utils


def DQN_train(exp_name, env_name, seed, render,
              params_dqn,
              params_policy,
              params_optim,
              params_nn, logdir=None):

    # Configure output directory for logging
    from external import logz
    logz.configure_output_dir(logdir)

    # Log experiments parameters
    params_extracted = {"exp_name": exp_name, "env_name": env_name, "seed":
    seed}

    for params in (params_nn, params_optim, params_policy, params_dqn):
        p = dict(params._asdict())
        params_extracted.update(p)
    logz.save_params(params_extracted)

    env = gym.make(env_name).unwrapped

    if params_optim.episode_len > 0:
        env = gym.make(env_name)
        env._max_episode_steps = params_optim.episode_len

    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space) == gym.spaces.Discrete

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    # Observation and action dimensions
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n

    # Creating Q-function approximator of our choise
    target_net = utils.QAproximator(ob_dim, ac_dim, params_nn.layer_size,
                                    params_nn.layer_num)
    policy_net = utils.QAproximator(ob_dim, ac_dim, params_nn.layer_size,
                                    params_nn.layer_num)

    # Replay buffer for storing our experiences from previous rollouts
    replay_buffer = utils.ReplayBuffer(params_dqn.replay_buffer_size)

    # Optizer - place where whole magic happen
    optimizer = utils.Optimizer(policy_net, target_net, replay_buffer,
                                params_dqn, params_optim)

    # Action chooser - selects action based on strategy and current policy
    action_chooser = utils.ActionChooser(policy_net, *params_policy)


    episode_durations = []
    log_rewards = 0

    timesteps = 0
    time_start = time.time()
    render_this_episode = False
    for e in range(params_optim.episode_num):
        state = torch.tensor([env.reset()], dtype=torch.float32)  # dim (1x4):

        log_rewards = 0
        timesteps_this_episode = 0
        if render and e % 100==0:
            render_this_episode = True
        else:
            render_this_episode = False

        while True:
            if render_this_episode:
                env.render()

            action = action_chooser.epsilon_greedy_policy(state, timesteps)
            #action = action_chooser.normal_policy(state, timesteps)

            next_state, reward, done, _ = env.step(action.item())

            next_state = torch.tensor(
                [next_state], dtype=torch.float32)  # dim (1x4)
            reward = torch.tensor([reward], dtype=torch.float32)  # dim (1)

            # negative reward when case of being in terminal state
            if done:
                #state_next = None
                reward = torch.tensor([0], dtype=torch.float32)

            log_rewards += reward.item()

            replay_buffer.push(state, action, next_state, reward)

            loss = optimizer.step(timesteps)

            state = next_state
            timesteps_this_episode += 1
            timesteps += 1



            if  done:
                episode_durations.append(timesteps_this_episode)
                #plot_durations(episode_durations)

                logz.log_tabular("Iteration", timesteps)
                logz.log_tabular("Episode", e)
                logz.log_tabular("Time", time.time() - time_start)
                logz.log_tabular("EpisodeLen", timesteps_this_episode)
                loss = 0 if loss is None else loss.item()
                logz.log_tabular("Loss", loss)
                logz.log_tabular("reward", log_rewards)
                logz.dump_tabular()
                break
        if e % 500 == 0:
            torch.save(target_net.state_dict(), logdir +"_target_" + str(e))
            torch.save(policy_net.state_dict(), logdir +"_policy_" + str(e))




def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated



if __name__ == "__main__":

    ENV_NAME = "MountainCar-v0"
    EXP_NAME = "DEV"

    # DQN default parameters
    USE_DOUBLE_DQN = False
    USE_POLYAK_AVERAGING = False
    TARGET_UPDATE_FREQ = 1
    REPLAY_BUFFER_SIZE = 10000
    GAMMA = 0.99
    POLYAK_INTERPOLATION_RATE = 0.99

    # Optimalize default parameters
    USE_GRADIENT_CLIPPING = False
    LEARNING_RATE = 0.001
    STEP_LR = 300
    BATCH_SIZE = 64
    EPISODE_NUM = 1000
    EPISODE_LEN = 200  # if value -1, episodes length are set to environment default

    # Policy evaluation default parameters
    EPS_GREEDY_START = 0.9
    EPS_GREEDY_END = 0.05
    EPS_GREEDY_DECAY = 200

    # Neutral network default parameters
    LAYER_NUM = 1
    LAYER_SIZE = 64
    USE_BATCHNORM = False
    SKIP_NONLINEARITY = False

    import argparse
    parser = argparse.ArgumentParser()

    # Experiment parameters
    parser.add_argument('--env_name', type=str, default=ENV_NAME)
    parser.add_argument('--exp_name', type=str, default=EXP_NAME)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)

    # DQN parameters
    parser.add_argument(
        '--use_double_dqn', action='store_true', default=USE_DOUBLE_DQN)
    parser.add_argument(
        '--use_polyak', action='store_true', default=USE_POLYAK_AVERAGING)
    parser.add_argument(
        '--target_update_freq', '-tuf', type=int, default=TARGET_UPDATE_FREQ)
    parser.add_argument(
        '--replay_buffer_size', '-rps', type=int, default=REPLAY_BUFFER_SIZE)
    parser.add_argument('--gamma', '-g', type=float, default=GAMMA)
    parser.add_argument('--polyak_interpolation_rate', '-pir', type=float, default=POLYAK_INTERPOLATION_RATE)

    # Optimazer parameters
    parser.add_argument(
        '--learning_rate', '-lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--step_lr', type=float, default=STEP_LR)
    parser.add_argument('--batch_size', '-b', type=int, default=BATCH_SIZE)
    parser.add_argument('--episode_num', '-epn', type=int, default=EPISODE_NUM)
    parser.add_argument('--episode_len', '-epl', type=int, default=EPISODE_LEN)
    parser.add_argument(
        '--use_gradient_clipping', action='store_true', default=USE_GRADIENT_CLIPPING)

    # Policy evaluation parameters
    parser.add_argument(
        '--eps_greedy_start', type=float, default=EPS_GREEDY_START)
    parser.add_argument('--eps_greedy_end', type=float, default=EPS_GREEDY_END)
    parser.add_argument(
        '--eps_greedy_decay', type=float, default=EPS_GREEDY_DECAY)

    # Neutral Network parameters
    parser.add_argument('--layer_num', '-ln', type=int, default=LAYER_NUM)
    parser.add_argument('--layer_size', '-ls', type=int, default=LAYER_SIZE)
    parser.add_argument(
        '--use_batchnorm', action='store_true', default=USE_BATCHNORM)
    parser.add_argument(
        '--skip_nonlinearity', action='store_true', default=SKIP_NONLINEARITY)
    args = parser.parse_args()

    ParamsDQN = namedtuple(
        "ParamsDQN",
        ("use_double_dqn", "use_polyak_averaging", "target_update_freq",
         "replay_buffer_size", "decay_rate", "polyak_interpolation_rate"))

    ParamsOptim = namedtuple(
        "ParamsOptim",
        ("learning_rate", "batch_size", "episode_num", "episode_len",
         "use_gradient_clipping", "step_lr"))

    ParamsPolicy = namedtuple(
        "ParamsPolicy",
        ("eps_greedy_start", "eps_greedy_end", "eps_greedy_decay"))

    ParamsNN = namedtuple(
        "ParamsNN",
        ("layer_num", "layer_size", "use_batchnorm", "skip_nonlinearity"))

    params_dqn = ParamsDQN(args.use_double_dqn, args.use_polyak,
                           args.target_update_freq, args.replay_buffer_size,
                           args.gamma, args.polyak_interpolation_rate)

    params_optim = ParamsOptim(args.learning_rate, args.batch_size,
                               args.episode_num, args.episode_len,
                               args.use_gradient_clipping, args.step_lr)

    params_policy = ParamsPolicy(args.eps_greedy_start, args.eps_greedy_end,
                                 args.eps_greedy_decay)

    params_nn = ParamsNN(args.layer_num, args.layer_size, args.use_batchnorm,
                         args.skip_nonlinearity)



    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime(
        "%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    for e in range(args.n_experiments):
        seed = args.seed + 10 * e

        print("Running experiment \"{}\" with seed: {} ".format(
            args.exp_name, seed))

        DQN_train(args.exp_name, args.env_name, seed, args.render, params_dqn, params_policy, params_optim,
                  params_nn, logdir=os.path.join(logdir, '%d' % seed))
