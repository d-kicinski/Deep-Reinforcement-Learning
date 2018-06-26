import gym
env_name = "CartPole-v0"
env = gym.make(env_name)

env = gym.wrappers.Monitor(env, "./video/", force=True, video_callable=lambda episode_id: episode_id%10==0)

for i_episode in range(51):
    observation = env.reset()
    to_render = (i_episode%10==0)
    for t in range(100):
        if to_render: env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
