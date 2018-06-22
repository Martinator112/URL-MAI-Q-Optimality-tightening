from networks import DuelingQNetworks
import gym
import numpy as np
import tensorflow as tf
import random
from collections import deque


  # epsilon_start = 1
  # epsilon_end = 0.05
  # epsilon_decay_length = 1e5		# number of steps over which to linearly decay epsilon
  # epsilon_decay_exp = 0.97	# exponential decay rate after reaching epsilon_end (per episode)
  # epsilon_linear_step = (epsilon_start - epsilon_end) / epsilon_decay_length

  # loaded_model = loadModel(saver, sess, './dqn_model_saved_3')
  # if loaded_model:
  #   total_steps = tf_total_steps.eval()
  #   epsilon = tf_epsilon.eval()
  #   episode = tf_episodes.eval()
  #   print("{} {} {}"$)
  # else:
  #   total_steps = 0
  #   epsilon = epsilon_start
  #   episode = 0


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def loadModel(saver, session, path):
  ckpt = tf.train.get_checkpoint_state(path)

  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(session, ckpt.model_checkpoint_path)
    print('Checkpoint restored')
    return True
  else:
    print('No checkpoint found')
    return False


# import matplotlib.pyplot as plt

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
update_slow_target_every = 100	# number of steps to use slow target as target before updating it to latest weights
train_every = 1			# number of steps to run the policy (and collect experience) before updating network weights
num_episodes = 5000		# number of episodes

log_every = 10 # log for tensorboard
save_every = 200 # save model parameters for restoring

replay_memory_capacity = int(1e6)	# capacity of experience replay memory
experience_replay = deque(maxlen=replay_memory_capacity)

tf.reset_default_graph()

tf_episode = tf.placeholder(dtype = tf.float32, name='episode')
tf_total_reward = tf.placeholder(dtype = tf.float32, name='total_episode_reward')
tf_episode_steps = tf.placeholder(dtype = tf.float32, name='episode_steps')



tf.summary.scalar('episode', tf_episode)
tf.summary.scalar('total reward', tf_total_reward)
tf.summary.scalar('episode steps', tf_episode_steps)

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

  epsilon_start = 1
  epsilon_end = 0.05
  epsilon_decay_length = 1e5		# number of steps over which to linearly decay epsilon
  epsilon_decay_exp = 0.97	# exponential decay rate after reaching epsilon_end (per episode)
  epsilon_linear_step = (epsilon_start - epsilon_end) / epsilon_decay_length

  loaded_model = loadModel(saver, sess, './dqn_model_saved_3')
  if loaded_model:
    total_steps = tf_total_steps.eval()
    epsilon = tf_epsilon.eval()
    episode = tf_episodes.eval()
    print("{} {} {}".format(total_steps,episode, epsilon ) )
  else:
    total_steps = 0
    epsilon = epsilon_start
    episode = 0

  while episode < num_episodes:
    total_reward = 0.0
    episode_steps = 0.0

    observation = env.reset()

    done = False

    while not done and episode_steps < max_episode_steps:
      if np.random.random() < epsilon:
        action = np.random.randint(actions_count)
      else:
        action = networks.choose_best_action(sess, observation)

      next_observation, reward, done, _ = env.step(action)

      is_not_terminal_state = 0.0 if done else 1.0
      experience_replay.append([observation, action, reward, next_observation, is_not_terminal_state])

      if total_steps % update_slow_target_every == 0:
        _ = networks.update_slow_target_network(sess)

      if total_steps % train_every == 0 and minibatch_size < len(experience_replay):
        minibatch = random.sample(experience_replay, minibatch_size)

        _ = networks.train(sess, minibatch)

      observation = next_observation
      total_reward += reward
      episode_steps += 1
      total_steps += 1

      if total_steps < epsilon_decay_length:
        epsilon -= epsilon_linear_step
      elif done:
        epsilon *= epsilon_decay_exp
      
      if done:
        networks.increment_episode(sess)

    if episode % log_every == 0:
      summary = sess.run([merged], feed_dict={
        tf_episode: episode,
        tf_total_reward: total_reward,
        tf_episode_steps: episode_steps
      })
      train_writer.add_summary(summary[0], log_iteration)
      log_iteration += 1

    if len(last_two_hundred_episode_rewards) == 200:
      print('Last two hundred episodes total reward: %7.3f'%(np.array(last_two_hundred_episode_rewards).mean()))
      last_two_hundred_episode_rewards = last_two_hundred_episode_rewards[1:]
      last_two_hundred_episode_rewards.append(total_reward)
    else:
      last_two_hundred_episode_rewards.append(total_reward)
      print('Last %2i episodes total reward: %7.3f'%(len(last_two_hundred_episode_rewards), np.array(last_two_hundred_episode_rewards).mean()))

    if (episode + 1) % save_every == 0:
      assign_epsilon_op = tf.assign(tf_epsilon, epsilon)
      assign_episodes_op = tf.assign(tf_episodes, episode)
      assign_total_steps_op = tf.assign(tf_total_steps, total_steps)
      sess.run([assign_epsilon_op, assign_episodes_op, assign_total_steps_op])

      print("{} {} {}".format(tf_epsilon.eval(), tf_episodes.eval(), tf_total_steps.eval()))

      saver.save(sess, './dqn_model_saved_3/', global_step=episode)

    print('Episode %2i, Reward: %7.3f, Steps: %i, Next eps: %7.3f'%(episode, total_reward, episode_steps, epsilon))
    episode += 1

