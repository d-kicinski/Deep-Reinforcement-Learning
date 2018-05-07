import numpy as np
import tensorflow as tf
import gym

import logz
import os
import time
import inspect
from multiprocessing import Process


def build_mlp(
        input_placeholder,
        output_size,
        scope,
        n_layers=2,
        size=64,
        activation=tf.tanh,
        output_activation=None):

    with tf.variable_scope(scope):
        # YOUR_CODE_HERE

        if n_layers == 0:
            return tf.layers.dense(inputs=input_placeholder,
                                   units=output_size,
                                   activation=output_activation)

        output = tf.layers.dense(inputs=input_placeholder,
                                 units=size,
                                 activation=activation)

        for l in range(n_layers - 1):
            output = tf.layers.dense(inputs=output,
                                     units=size,
                                     activation=activation)

        output = tf.layers.dense(inputs=output,
                                 units=output_size,
                                 activation=output_activation)

        return output



def pathlength(path):
    return len(path["reward"])



def collect_paths(sess, sy_sampled_ac, sy_ob_no, env, min_timesteps,
                  max_path_length, animate, itr):
    """Collects one batch of dataset. Return that many paths that when summed
       contain at minimum 'min_timesteps' timesteps.

    Returns:
    paths: dict of: "observation" , "reward", "action" mapped to ndarrays.
    timesteps_this_batch: int - number of collected timesteps

    """

    timesteps_this_batch = 0
    paths = []
    while True:
        ob = env.reset()
        obs, acs, rewards = [], [], []
        animate_this_episode = (len(paths) == 0 and (itr % 10 == 0) and animate)
        steps = 0
        while True:
            if animate_this_episode:
                env.render()
                time.sleep(0.05)
            obs.append(ob)
            ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no: ob[None]})
            ac = ac[0][0]
            acs.append(ac)
            ob, rew, done, _ = env.step(ac)
            rewards.append(rew)
            steps += 1
            if done or steps > max_path_length:
                break

        path = {"observation": np.array(obs),
                "reward": np.array(rewards),
                "action": np.array(acs)}
        paths.append(path)

        timesteps_this_batch += pathlength(path)

        if timesteps_this_batch > min_timesteps:
            break

    return paths, timesteps_this_batch



def get_reward(paths, gamma=0.99, reward_to_go=True):
    """Computing Q-values

    Your code should construct numpy arrays for Q-values which will be used to compute
    advantages (which will in turn be fed to the placeholder you defined above).

    Recall that the expression for the policy gradient PG is

    PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * (Q_t - b_t )]

    where

    tau=(s_0, a_0, ...) is a trajectory,
    Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
    and b_t is a baseline which may depend on s_t.

    You will write code for two cases, controlled by the flag 'reward_to_go':

    Case 1: trajectory-based PG
    (reward_to_go = False)

    Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over
    entire trajectory (regardless of which time step the Q-value should be for).

    For this case, the policy gradient estimator is

    E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]

    where
    Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.

    Thus, you should compute
    Q_t = Ret(tau)

    Case 2: reward-to-go PG
    (reward_to_go = True)

    Here, you estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting
    from time step t. Thus, you should compute

    Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}

    Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
    like the 'ob_no' and 'ac_na' above.


    Parameters
    ----------
    paths
    gamma
    reward_to_go

    Returns
    -------

    """

    if reward_to_go:
        #  Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}

        q_values_n = []
        for path in paths:
            path_rewards = path["reward"]
            q_values = []  # q_values for current path

            # precompute vector with dicounted gamma
            gammas = np.array([gamma ** i for i in range(len(path_rewards))])

            # compute q_value for each time step in trajectory
            for i in range(len(path_rewards)):
                if i == 0:
                    q_values.append(gammas @ path_rewards)
                else:
                    q_values.append(gammas[:-i] @ path_rewards[i:])

            q_values_n.append(q_values)

        q_n = np.concatenate(q_values_n)
    else:
        #  Q_t = sum_{t'=0}^T gamma^t' r_{t'}.

        reward_sum = 0  # sum of reward for each single trajectory
        q_values_n = []
        for path in paths:
            path_rewards = path["reward"]
            for i, reward in enumerate(path_rewards):
                reward_sum += reward * gamma ** i
            q_values_n.append([reward_sum] * len(path_rewards))
        q_n = np.concatenate(q_values_n)

    return q_n



def train_PG(exp_name='',
             env_name='CartPole-v0',
             n_iter=100,
             gamma=1.0,
             min_timesteps=1000,
             max_path_length=None,
             learning_rate=5e-3,
             reward_to_go=True,
             animate=True,
             logdir=None,
             normalize_advantages=True,
             nn_baseline=False,
             seed=0,
             # network arguments
             n_layers=1,
             size=32
             ):

    start = time.time()

    # Configure output directory for logging
    logz.configure_output_dir(logdir)

    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Make the gym environment
    env = gym.make(env_name)

    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps


    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]


    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
    if discrete:
        sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
    else:
        sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac",
                                  dtype=tf.float32)

    # Define a placeholder for advantages
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)

    # ========================================================================================#
    #                           ----------SECTION 4----------
    # Networks
    #
    # Make symbolic operations for
    #   1. Policy network outputs which describe the policy distribution.
    #       a. For the discrete case, just logits for each action.
    #
    #       b. For the continuous case, the mean / log std of a Gaussian distribution over
    #          actions.
    #
    #      Hint: use the 'build_mlp' function you defined in utilities.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ob_no'
    #
    #   2. Producing samples stochastically from the policy distribution.
    #       a. For the discrete case, an op that takes in logits and produces actions.
    #
    #          Should have shape [None]
    #
    #       b. For the continuous case, use the reparameterization trick:
    #          The output from a Gaussian distribution with mean 'mu' and std 'sigma' is
    #
    #               mu + sigma * z,         z ~ N(0, I)
    #
    #          This reduces the problem to just sampling z. (Hint: use tf.random_normal!)
    #
    #          Should have shape [None, ac_dim]
    #
    #      Note: these ops should be functions of the policy network output ops.
    #
    #   3. Computing the log probability of a set of actions that were actually taken,
    #      according to the policy.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ac_na', and the
    #      policy network output ops.
    #
    # ========================================================================================#

    if discrete:
        # Network takes current observated state outputs log probs of action to take
        sy_logits_na = build_mlp(
            input_placeholder=sy_ob_no,
            output_size=ac_dim,
            scope="ham",
            n_layers=n_layers,
            size=size,
            activation=tf.nn.relu,
        )

        # Sample some action from distribution outputed by network
        # used for policy evaluation!

        sy_sampled_ac = tf.multinomial(sy_logits_na, 1)

        # Get log-prob of actual taken action in trajectiory from net outputs
        # indices = tf.transpose([tf.range(sy_ac_na.shape[0]), sy_ac_na])
        indices = tf.one_hot(sy_ac_na, depth=2, dtype=tf.int32)
        sy_logprob_n = tf.gather_nd(sy_logits_na, indices)

    else:
        # Network takes current observated state and outputs the meand and log
        # std of Gaussion distribution over actions.

        net_output_dim = ac_dim
        sy_mean = build_mlp(
            input_placeholder=sy_ob_no,
            output_size=net_output_dim,
            scope="ham",
            n_layers=n_layers,
            size=size,
            activation=tf.nn.relu,
        )

        # logstd should just be a trainable variable, not a network output.
        sy_logstd = tf.Variable(tf.ones(ac_dim))

        sy_sampled_ac = sy_mean + sy_logstd * tf.random_normal(shape=[None, ac_dim])

        # Hint: Use the log probability under a multivariate gaussian.
        dist = tf.distributions.Normal(loc=[sy_mean], scale=[sy_logstd])
        sy_logprob_n = tf.log(dist.prob(sy_ac_na))


    # Loss Function and Training Operation
    negative_likelihoods = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(sy_ac_na, depth=ac_dim), logits=sy_logits_na)
    weighted_negative_likelihoods = tf.multiply(negative_likelihoods, sy_adv_n)

    loss = tf.reduce_mean(weighted_negative_likelihoods)

    update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    if nn_baseline:
        baseline_prediction = tf.squeeze(build_mlp(
            sy_ob_no,
            1,
            "nn_baseline",
            n_layers=n_layers,
            size=size))
        # Define placeholders for targets, a loss function and an update op for fitting a
        # neural network baseline. These will be used to fit the neural network baseline.
        # YOUR_CODE_HERE
        # baseline_update_op = TODO
        raise NotImplemented

    # ========================================================================================#
    # Tensorflow Engineering: Config, Session, Variable initialization
    # ========================================================================================#

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1,
                               intra_op_parallelism_threads=1)
    sess = tf.Session(config=tf_config)
    sess.__enter__()  # equivalent to `with sess:`
    tf.global_variables_initializer().run()  # pylint: disable=E1101

    # ========================================================================================#
    # Training Loop
    # ========================================================================================#

    total_timesteps = 0

    for itr in range(n_iter):
        print("********** Iteration %i ************" % itr)

        # Collect paths until we have enough timesteps for one batch
        paths, num_collected_timesteps = collect_paths(sess, sy_sampled_ac, sy_ob_no, env,
                                                       min_timesteps, max_path_length,
                                                       animate, itr)
        total_timesteps += num_collected_timesteps

        # Build arrays for observation, action for the policy gradient update
        #  by concatenating  across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        q_n = get_reward(paths, gamma, reward_to_go)

        if nn_baseline:
            # If nn_baseline is True, use your neural network to predict reward-to-go
            # at each timestep for each trajectory, and save the result in a variable 'b_n'
            # like 'ob_no', 'ac_na', and 'q_n'.
            #
            # Hint #bl1: rescale the output from the nn_baseline to match the statistics
            # (mean and std) of the current or previous batch of Q-values. (Goes with Hint
            # #bl2 below.)

            # fuckme : what the fuck?
            # b_n = TODO
            # adv_n = q_n - b_n
            pass
        else:
            adv_n = q_n.copy()

        if normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1.
            adv_n = (adv_n - np.mean(adv_n)) / np.std(adv_n)

        if nn_baseline:
            # If a neural network baseline is used, set up the targets and the inputs for the
            # baseline.
            #
            # Fit it to the current batch in order to use for the next iteration. Use the
            # baseline_update_op you defined earlier.
            #
            # Hint #bl2: Instead of trying to target raw Q-values directly, rescale the
            # targets to have mean zero and std=1. (Goes with Hint #bl1 above.)

            # fuckme part two
            # YOUR_CODE_HERE
            pass

        #print("ob: ", type(ob_no))
        #print("na: ", type(ob_no))
        #print("adv: ", type(ob_no))


        loss_before = sess.run(loss, feed_dict={
            sy_ob_no: ob_no,  # observation
            sy_ac_na: ac_na,  # taken actions
            sy_adv_n: adv_n  # adventages
        })

        sess.run(update_op, feed_dict={
            sy_ob_no: ob_no,  # observation
            sy_ac_na: ac_na,  # taken actions
            sy_adv_n: adv_n  # adventages
        })

        loss_after = sess.run(loss, feed_dict={
            sy_ob_no: ob_no,  # observation
            sy_ac_na: ac_na,  # taken actions
            sy_adv_n: adv_n  # adventages
        })

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]

        logz.log_tabular("Loss_before", loss_before)
        logz.log_tabular("Loss_after", loss_after)

        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", len(ac_na))
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.pickle_tf_vars()



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna',
                        action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)
    args = parser.parse_args()

    if not (os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime(
        "%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    for e in range(args.n_experiments):
        seed = args.seed + 10 * e
        print('Running experiment with seed %d' % seed)

        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir, '%d' % seed),
                normalize_advantages=not (args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline,
                seed=seed,
                n_layers=args.n_layers,
                size=args.size
            )

        # Awkward hacky process runs, because Tensorflow does not like
        # repeatedly calling train_PG in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        p.join()




if __name__ == "__main__":
    # main()
    # build_mlp_test()

    train_PG(
        n_iter=100
    )


def build_mlp_test():
    with tf.variable_scope("test") as test_scope:
        input_holder = tf.placeholder(tf.float32, [None, 20])

        output = build_mlp(
            input_placeholder=input_holder,
            output_size=10,
            scope=test_scope,
            n_layers=0,
            size=64,
            activation=tf.nn.relu,
            output_activation=None
        )

        print(output)
