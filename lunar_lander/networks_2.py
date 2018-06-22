# def generate_network(s, trainable, reuse):
# 	hidden = tf.layers.dense(s, h1, activation = tf.nn.relu, trainable = trainable, name = 'dense', reuse = reuse)
# 	hidden_drop = tf.layers.dropout(hidden, rate = dropout, training = trainable & is_training_ph)
# 	hidden_2 = tf.layers.dense(hidden_drop, h2, activation = tf.nn.relu, trainable = trainable, name = 'dense_1', reuse = reuse)
# 	hidden_drop_2 = tf.layers.dropout(hidden_2, rate = dropout, training = trainable & is_training_ph)
# 	hidden_3 = tf.layers.dense(hidden_drop_2, h3, activation = tf.nn.relu, trainable = trainable, name = 'dense_2', reuse = reuse)
# 	hidden_drop_3 = tf.layers.dropout(hidden_3, rate = dropout, training = trainable & is_training_ph)
# 	action_values = tf.squeeze(tf.layers.dense(hidden_drop_3, n_actions, trainable = trainable, name = 'dense_3', reuse = reuse))
# 	return action_values
import tensorflow as tf
import numpy as np

# parameters = {
#   'learning_rate': ,
#   'learning_rate_decay':,
#   'dropout': ,
#   'hidden_l1_size': ,
#   'hidden_l2_size': ,
#   'hidden_l3_size': ,
#   'actions_count': ,
#   'gamma': ,
#   'state_dimensions': ,
#   'minibatch_size': ,
# }

class DuelingQNetworks():
  def __init__(self, parameters):
    self.learning_rate = parameters["learning_rate"]
    self.learning_rate_decay = parameters["learning_rate_decay"]
    self.dropout = parameters["dropout"]

    self.hidden_l1_size = parameters["hidden_l1_size"]
    self.hidden_l2_size = parameters["hidden_l2_size"]
    self.hidden_l3_size = parameters["hidden_l3_size"]

    self.actions_count = parameters["actions_count"]
    self.gamma = parameters["gamma"]
    self.minibatch_size = parameters["minibatch_size"]

    self.state_dim = parameters["state_dimensions"]

    self.state = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim])
    self.action_input = tf.placeholder(dtype=tf.int32, shape=[None])
    self.reward_input = tf.placeholder(dtype=tf.float32, shape=[None])
    self.is_training_phase = tf.placeholder(dtype=tf.bool, shape=())
    self.is_not_terminal = tf.placeholder(dtype=tf.float32, shape=[None])

    self.transition_length = 4

    self.forward_states = []
    self.backward_states = []
    self.forward_action_values = []
    self.backward_action_values = []
    self.forward_rewards = []
    self.backward_rewards = []

    for i in np.arange(self.transition_length):
      self.forward_states[i] = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim])
      self.forward_rewards[i] = tf.placeholder(dtype=tf.float32, shape=[self.transition_length])
      self.backward_states[i] = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim])
      self.backward_rewards[i] = tf.placeholder(dtype=tf.float32, shape=[self.transition_length])

    self.episodes = tf.Variable(0.0, trainable=False, name='episodes')
    self.episode_inc_op = self.episodes.assign_add(1)

    with tf.variable_scope('current_network') as scope:
      self.action_values = self.build_network(self.state, frozen_network=False, trainable=True, reuse=False)

      for i in np.arange(self.transition_length):
        self.forward_action_values[i] = self.build_network(self.forward_states[i], frozen_network=True, trainable=False, reuse=True)
      
      for i in np.arange(self.transition_length):
        self.backward_action_values[i] = self.build_network(self.backward_states[i], frozen_network=True, trainable=False, reuse=True)

    with tf.variable_scope('slow_target_network', reuse = False) as frozen_scope:
      for i in np.arange(self.transition_length):
        self.forward_slow_target_action_values[i] = self.build_network(self.forward_states[i], frozen_network=True, trainable=False, reuse=False)
      
      for i in np.arange(self.transition_length):
        self.backward_slow_target_action_values[i] = self.build_network(self.backward_states[i], frozen_network=True, trainable=False, reuse=False)

    q_network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='current_network')
    slow_target_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_network')

    # update values for slowly-changing target network to match current critic network
    update_slow_target_ops = []
    for i, slow_target_var in enumerate(slow_target_network_vars):
      update_slow_target_op = slow_target_var.assign(q_network_vars[i])
      update_slow_target_ops.append(update_slow_target_op)

    update_slow_target_op = tf.group(*update_slow_target_ops, name='update_slow_target')
    self.update_slow_target_op = update_slow_target_op

        
    difference = self.compute_target_actions_values() - self.estimate_target_actions_taken()
    loss = tf.reduce_mean(tf.square(difference))

    for var in q_network_vars:
	    if not 'bias' in var.name:
		    loss += parameters["l2_regularization_speed"] * 0.5 * tf.nn.l2_loss(var)

    self.train_op = tf.train.AdamOptimizer(self.learning_rate * self.learning_rate_decay ** self.episodes).minimize(loss)


  def build_network(self, state, frozen_network = False, trainable = True, reuse = False):
    self.hidden_l1 = tf.layers.dense(state, self.hidden_l1_size, activation = tf.nn.relu, trainable = trainable, name = 'hidden_l1', reuse = reuse)
    self.dropout_l1 = tf.layers.dropout(self.hidden_l1, training=trainable & self.is_training_phase, rate = self.dropout)
    self.hidden_l2 = tf.layers.dense(self.dropout_l1, self.hidden_l2_size, activation=tf.nn.relu, trainable=trainable, name = 'hidden_l2', reuse = reuse)
    self.dropout_l2 = tf.layers.dropout(self.hidden_l2, training=trainable & self.is_training_phase, rate = self.dropout)
    self.hidden_l3 = tf.layers.dense(self.dropout_l2, self.hidden_l3_size, trainable=trainable, activation=tf.nn.relu, name = 'hidden_l3', reuse = reuse)
    self.dropout_l3 = tf.layers.dropout(self.hidden_l3, training=trainable & self.is_training_phase, rate = self.dropout)

    self.final_layer = tf.layers.dense(self.dropout_l3, self.actions_count, activation=tf.nn.relu, trainable=trainable, name="final", reuse = reuse)
    action_values = tf.squeeze(self.final_layer)

    if frozen_network:
      action_values = tf.stop_gradient(action_values)

    return action_values

  def choose_best_actions(self, i):
    return tf.cast(tf.argmax(self.forward_action_values[i], axis=1), tf.int32)

  def compute_slow_target_action_values(self, i, next_action_indices):
    minibatch_indices = tf.range(self.minibatch_size, name = 'slow_targets_batch_action_positions')
    next_action_values_indices = tf.stack((minibatch_indices, next_action_indices), axis=1, name = 'slow_targets_batch_position_action_index_pairs')
    return tf.gather_nd(self.forward_slow_target_action_values[i], next_action_values_indices, name = "slow_target_batch_gathered_values")

  def estimate_target_actions_taken(self):
    minibatch_indices = tf.range(self.minibatch_size, name = 'estimated_targets_batch_action_positions')
    action_values_indices = tf.stack((minibatch_indices, self.action_input), axis = 1, name = 'estimated_targets_batch_position_action_index_pairs')
    return tf.gather_nd(self.action_values, action_values_indices, name = 'estimated_targets_batch_gathered_values')

  def compute_target_actions_values(self):
    next_targets = self.forward_rewards[0] + self.gamma * self.compute_slow_target_action_values(0, self.choose_best_actions(0))

    violations_penalties = tf.identity(next_targets)

    forward_targets = np.zeros(self.transition_length)
    backward_targets = np.zeros(self.transition_length)
    for i in np.arange(1, self.transition_length):
      forward_targets[i] = self.forward_rewards[i] + (self.gamma ** i) * self.compute_slow_target_action_values(i, self.choose_best_actions(i)) 
    
    for i in np.arange(0, self.transition_length):
      backward_targets[i] = self.backward_rewards[i] + (self.gamma ** i) * self.compute_slow_target_action_values(i, self.choose_best_actions(i)) 

    max_q_value_indices = tf.argmax(backward_targets, axis = 1)
    min_q_value_indices = tf.argmax(forward_targets, axis = 1)

    flattened_forward_targets = tf.reshape(forward_targets, [-1])
    flattened_backward_targets = tf.reshape(backward_targets, [-1])

    flattened_backward_ind_max = max_q_value_indices + tf.cast(tf.range(tf.shape(backward_targets)[0]) * tf.shape(backward_targets)[1], tf.int64)
    flattened_forward_ind_max = min_q_value_indices + tf.cast(tf.range(tf.shape(forward_targets)[0]) * tf.shape(forward_targets)[1], tf.int64)

    min_q_values = tf.gather(flattened_forward_targets, flattened_forward_ind_max)
    max_q_values = tf.gather(flattened_backward_targets, flattened_backward_ind_max)

    upper_violation = tf.where(tf.greater(max_q_values - 20, next_targets))
    lower_violation = tf.where(tf.greater(next_targets, min_q_values + 20))

    upper_violation_range = tf.range(tf.shape(upper_violation)[0])
    lower_violation_range = tf.range(tf.shape(lower_violation)[0])

    flattened_upper_violation = tf.gather_nd(upper_violation, tf.stack((upper_violation_range, tf.fill([tf.shape(upper_violation)[0]], 0)), axis = 1))
    flattened_lower_violation = tf.gather_nd(lower_violation, tf.stack((lower_violation_range, tf.fill([tf.shape(lower_violation)[0]], 0)), axis = 1))

    diff = tf.setdiff1d(flattened_upper_violation, flattened_lower_violation)
    diff_2 = tf.setdiff1d(flattened_lower_violation, flattened_upper_violation)
    diff_3 = tf.setdiff1d(flattened_upper_violation, diff[0])

    only_upper_bound_violation_indices = tf.stack((diff[0], tf.fill(tf.shape(diff[0]), tf.constant(0, dtype=tf.int64))), axis = 1)
    only_lower_bound_violation_indices = tf.stack((diff_2[0], tf.fill(tf.shape(diff_2[0]), tf.constant(0, dtype=tf.int64))), axis = 1)
    both_bounds_violations_indices = tf.stack((diff_3[0], tf.fill(tf.shape(diff_3[0]), tf.constant(0, dtype=tf.int64))), axis = 1)

    violations_penalties += tf.scatter_nd(only_upper_bound_violation_indices, tf.gather(max_q_values, only_upper_bound_violation_indices), [self.minibatch_size, 1])
    violations_penalties += tf.scatter_nd(only_lower_bound_violation_indices, tf.gather(min_q_values, only_lower_bound_violation_indices), [self.minibatch_size, 1])
    violations_penalties += tf.scatter_nd(both_bounds_violations_indices, 0.5 * tf.gather(min_q_values, both_bounds_violations_indices) + tf.gather(max_q_values, both_bounds_violations_indices), [self.minibatch_size, 1])

    final_targets = 0.8 * next_targets + 0.2 * violations_penalties

    return final_targets

  def compute_slow_target_action_values_forward_1(self, next_action_indices):
    minibatch_indices = tf.range(self.minibatch_size)
    next_action_values_indices = tf.stack((minibatch_indices, next_action_indices), axis=1)
    return tf.gather_nd(self.slow_target_action_values_forward_1, next_action_values_indices)

  def choose_best_action_values_forward_1(self):
    return tf.cast(tf.argmax(self.next_action_values_forward_1 , axis=1), tf.int32)

  def choose_best_action_values_forward_2(self):
    return tf.cast(tf.argmax(self.next_action_values_forward_2 , axis=1), tf.int32)

  def choose_best_action_values_forward_3(self):
    return tf.cast(tf.argmax(self.next_action_values_forward_3 , axis=1), tf.int32)


  def choose_best_action_values_backward_1(self):
    return tf.cast(tf.argmax(self.next_action_values_forward_1 , axis=1), tf.int32)

  def choose_best_action_values_forward_2(self):
    return tf.cast(tf.argmax(self.next_action_values_forward_2 , axis=1), tf.int32)

  def choose_best_action_values_forward_3(self):
    return tf.cast(tf.argmax(self.next_action_values_forward_3 , axis=1), tf.int32)

  def choose_best_action_values_forward_3(self):
    return tf.cast(tf.argmax(self.next_action_values_forward_3 , axis=1), tf.int32)


  def train(self, session, minibatch):
    return session.run(self.train_op, feed_dict = {
      self.state: np.asarray([elem[0] for elem in minibatch]),
      self.action_input: np.asarray([elem[1] for elem in minibatch]),
      self.reward_input: np.asarray([elem[2] for elem in minibatch]),
      self.next_state: np.asarray([elem[3] for elem in minibatch]),
      self.is_not_terminal: np.asarray([elem[4] for elem in minibatch]),
      self.is_training_phase: False
    })


  def choose_best_action(self, session, state):
    feed_dict = {
      # make it one dimension bigger
      self.state: state[None],
      self.is_training_phase: False
    }
    q_s = session.run(self.action_values, feed_dict)
    action = np.argmax(q_s)

    return action

  def increment_episode(self, session):
    _ = session.run(self.episode_inc_op)

  def update_slow_target_network(self, session):
    return session.run(self.update_slow_target_op)
