import balatro_gym # noqa
import gymnasium as gym
from balatro_gym import agent
from matplotlib import pyplot as plt
from gymnasium.wrappers import RecordVideo
import numpy as np
from tqdm import tqdm


env = gym.make("Balatro-v0", render_mode="rgb_array")
observation = env.reset()

training_period = 10000


learning_rate = 0.01
n_episodes = 100000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1


balatro_agent = agent.BalatroAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)
env = gym.wrappers.RecordEpisodeStatistics(env)
env = RecordVideo(env, video_folder="Balatro-agent", name_prefix="training",
                  episode_trigger=lambda x: x % training_period == 0)


for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        
        action = balatro_agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)


        balatro_agent.update(obs, action, reward, terminated, next_obs)

        done = terminated or truncated
        obs = next_obs

    balatro_agent.decay_epsilon()

env.close()


fig, axs = plt.subplots(1, 3, figsize=(20, 8))

# np.convolve will compute the rolling mean for 100 episodes

axs[0].plot(np.convolve(np.ravel(env.return_queue), np.ones(100)))
axs[0].set_title("Episode Rewards")
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Reward")

axs[1].plot(np.convolve(np.ravel(env.length_queue), np.ones(100)))
axs[1].set_title("Episode Lengths")
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Length")


plt.tight_layout()
plt.show()