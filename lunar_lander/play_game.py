from networks import DuelingQNetworks
import gym
import numpy as np
import tensorflow as tf
import random
from collections import deque
import time

def loadModel(saver, session, path):
  ckpt = tf.train.get_checkpoint_state(path)

  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(session, ckpt.model_checkpoint_path)
    print('Checkpoint restored')
    return True
  else:
    print('No checkpoint found')
    return False

env = gym.make('LunarLander-v2')

state_dimensions = np.prod(np.array(env.observation_space.shape))
actions_count = env.action_space.n
minibatch_size = 1024
parameters = {
  'learning_rate': 5e-5,
  'learning_rate_decay': 1,
  'dropout': 0,
  'hidden_l1_size': 512,
  'hidden_l2_size': 512,
  'hidden_l3_size': 512,
  'actions_count': actions_count,
  'gamma': 0.99,
  'state_dimensions': state_dimensions,
  'minibatch_size': minibatch_size,
  'l2_regularization_speed': 1e-6
}

max_episode_steps = 10000	# default max number of steps per episode (unless env has a lower hardcoded limit)
num_episodes = 5000		# number of episodes

tf.reset_default_graph()

log_iteration = 0

networks = DuelingQNetworks(parameters)

last_two_hundred_episode_rewards = []

with tf.Session() as sess:
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter('./train_3', sess.graph)

  tf_epsilon = tf.Variable(0.0, name = 'epsilon_policy')
  tf_episodes = tf.Variable(0, name = 'episodes')
  tf_total_steps = tf.Variable(0, name = 'total_steps')

  saver = tf.train.Saver(max_to_keep=20)

  sess.run(tf.global_variables_initializer())

  loaded_model = loadModel(saver, sess, './dqn_model_saved_3')
  if loaded_model:
    total_steps = tf_total_steps.eval()
    epsilon = tf_epsilon.eval()
    episode = tf_episodes.eval()
    print("{} {} {}".format(total_steps,episode, epsilon ) )

  while True:
    total_reward = 0.0
    episode_steps = 0.0

    observation = env.reset()

    done = False

    while not done and episode_steps < max_episode_steps:
      action = networks.choose_best_action(sess, observation)

      env.render()
      time.sleep(0.05)

      next_observation, reward, done, _ = env.step(action)
      observation = next_observation

      total_reward += reward
      episode_steps += 1
      total_steps += 1
      
    if len(last_two_hundred_episode_rewards) == 200:
      print('Last two hundred episodes total reward: %7.3f'%(np.array(last_two_hundred_episode_rewards).mean()))
      last_two_hundred_episode_rewards = last_two_hundred_episode_rewards[1:]
      last_two_hundred_episode_rewards.append(total_reward)
    else:
      last_two_hundred_episode_rewards.append(total_reward)
      print('Last %2i episodes total reward: %7.3f'%(len(last_two_hundred_episode_rewards), np.array(last_two_hundred_episode_rewards).mean()))

    print('Episode %2i, Reward: %7.3f, Steps: %i, Next eps: %7.3f'%(episode, total_reward, episode_steps, epsilon))
    episode += 1

