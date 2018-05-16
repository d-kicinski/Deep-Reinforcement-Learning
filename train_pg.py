import numpy as np
import tensorflow as tf
import gym

import logz
import os
import time
import inspect
from multiprocessing import Process


class BaselinePredictor(object):
    def __init__(self, sy_ob_no, epoch_num, learning_rate, n_layers, size):
        """Builds NN"""
        self.sy_ob_no = sy_ob_no
        self.epoch_num = epoch_num

        self.baseline_pred = tf.squeeze(build_mlp(
            input_placeholder = sy_ob_no,
            output_size=1,
            scope="nn_baseline",
            n_layers=n_layers,
            size=size))

        self.sy_target = tf.placeholder(shape=[None], name="nn_baseline/target",
                                    dtype=tf.float32)

        #loss = 0.5 * tf.reduce_mean((self.sy_target-self.baseline_pred)**2)
        loss = tf.nn.l2_loss(self.baseline_pred - self.sy_target)
        self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    def fit(self, inputs, labels):
        sess = tf.get_default_session()
        for _ in range(self.epoch_num):
            sess.run([self.update_op], feed_dict={
                self.sy_ob_no: inputs, self.sy_target: labels})

    def predict(self, inputs):
        sess = tf.get_default_session()
        return sess.run([self.baseline_pred], feed_dict={self.sy_ob_no:
                                                                 inputs})


def build_mlp(
        input_placeholder,
        output_size,
        scope,
        n_layers=2,
        size=64,
        activation=tf.tanh,
        output_activation=None):

    with tf.variable_scope(scope):

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
                  max_path_length, to_animate, itr, discrete):
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
        #animate_this_episode = (len(paths) == 0 and (itr % 10 == 0) and animate)

        animate = False
        to_animate.animate = (len(paths) == 0 and (itr % 20 == 0) and animate)

        steps = 0
        while True:
            #if animate_this_episode:
            if to_animate.animate:
                env.render()
                time.sleep(0.05)

            obs.append(ob)
            ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no: ob[None]})
            #print("ac from tensor: ", ac)

            acs.append(ac.flatten())
            if discrete:
                ob, rew, done, _ = env.step(ac.flatten()[0])
            else:
                ob, rew, done, _ = env.step(ac.flatten())

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
    """Computing Q-values"""

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

    return q_n.flatten()


class ToAnimate:
    def __init__(self, animate):
        self.animate = animate

    def __call__(self, episode_id):
        return self.animate


def train_PG(exp_name='',
             env_name='CartPole-v0',
             n_iter=100,
             gamma=1.0,
             min_timesteps=1000,
             max_path_length=None,
             learning_rate=5e-3,
             reward_to_go=True,
             to_animate=True,
             logdir=None,
             normalize_advantages=True,
             nn_baseline=False,
             seed=0,
             # network arguments
             n_layers=1,
             size=32,
             save_video=False
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
    #env._max_episode_steps = 4000



    to_animate = ToAnimate(False)
    to_animate.animate = False

    if save_video:
        env = gym.wrappers.Monitor(env, "./video/", force=True, video_callable=to_animate)


    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps


    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]


    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
    if discrete:
        #sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.int32)
        sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
    else:
        sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32)

    # Define a placeholder for advantages
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)


    if discrete:
        # Network takes current observated state outputs log probs of action to take
        sy_logits_na = build_mlp(
            input_placeholder=sy_ob_no,
            output_size=ac_dim,
            scope="policy",
            n_layers=n_layers,
            size=size,
            activation=tf.nn.relu,
        )

        # Sample some action from distribution outputed by network
        # used for policy evaluation!

        sy_sampled_ac = tf.multinomial(sy_logits_na, 1)

        # positive log probs of actual taken action, so the values are negative
        # Note thate cross entropy negate log probilities
        # FIXME: now this is only valid for just one discrete action
        sy_logprob_n = -tf.nn.sparse_softmax_cross_entropy_with_logits(
                    #labels=tf.reshape( sy_ac_na, [-1]), logits=sy_logits_na)
                    labels= sy_ac_na, logits=sy_logits_na)


    else:
        # Network takes current observated state and outputs the meand and log
        # std of Gaussion distribution over actions.

        net_output_dim = ac_dim
        sy_mean = build_mlp(
            input_placeholder=sy_ob_no,
            output_size=net_output_dim,
            scope="policy",
            n_layers=n_layers,
            size=size,
            activation=tf.nn.relu,
        )

        sy_logstd_na = tf.get_variable("policy/logstd", [ac_dim],
                                       initializer=tf.zeros_initializer(), dtype=tf.float32)
        dist = tf.distributions.Normal(loc=[sy_mean],
                                       scale=[tf.exp(sy_logstd_na)], validate_args=True)
        sy_sampled_ac = dist.sample()

        # later i'will want to maximize propbs in all dimensions so adding them now
        # nothing change really but generalize loss function
        sy_logprob_n = tf.reduce_sum(dist.log_prob(sy_ac_na), axis=1)


    # Loss Function and Training Operation

    # negate becaouse of minimalization instead maximalization
    loss = -tf.reduce_mean(sy_logprob_n * sy_adv_n)

    update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    if nn_baseline:
        baseline_predictor = BaselinePredictor(sy_ob_no, epoch_num=500,
                                               learning_rate=learning_rate,
                                               n_layers=n_layers, size=size)


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
    #
    total_timesteps = 0

    for itr in range(n_iter):
        print("********** Iteration %i ************" % itr)

        # Collect paths until we have enough timesteps for one batch
        paths, num_collected_timesteps = collect_paths(sess, sy_sampled_ac, sy_ob_no, env,
                                                       min_timesteps, max_path_length,
                                                       to_animate, itr, discrete)
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

            # Rescaling the output to mach statistics of Q-values
            b_n = baseline_predictor.predict(ob_no)[0]
            b_n = (b_n - np.mean(b_n)) / np.std(b_n)
            b_n = np.mean(q_n) + (b_n * np.std(q_n))
            adv_n = q_n - b_n

            # fuckme : what the fuck?  // few days later: calm down boi gatcha ya
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

            baseline_predictor.fit(inputs=ob_no, labels=(q_n - np.mean(q_n)) / np.std(q_n))

        if discrete: ac_na = ac_na.flatten()

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

        print(loss_before)
        #logz.log_tabular("Loss_before", loss_before)
        #logz.log_tabular("Loss_after", loss_after)
        logz.log_tabular("delta_loss", loss_after-loss_before)

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
                to_animate=args.render,
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
        #env_name="MountainCar-v0",
        #env_name = "MountainCarContinuous-v0",
        env_name="Pendulum-v0",
        #env_name="CartPole-v0",
        nn_baseline=True,
        normalize_advantages=True,
        n_iter=500,
        max_path_length=4000,
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
