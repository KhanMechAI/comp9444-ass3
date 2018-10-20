import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA =  .2# discount factor
INITIAL_EPSILON = .05# starting value of epsilon
FINAL_EPSILON = .001 # final value of epsilon
EPSILON_DECAY_STEPS =  100000# decay period
HIDDEN_UNITS = 20
HIDDEN_UNITS_2 = 10

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

# TODO: Define Network Graph

class DeepQNN:
    def __init__(self, state_size, action_size, learning_rate, name="DQN"):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            #Input place holder definitions
            self.inputs = tf.placeholder("float", [None, *state_size], name="inputs")
            self.actions = tf.placeholder("float", [None, * action_size], name="actions")
            self.q_target = tf.placeholder("float", [None], name="q_target")

            #Network architecture
            self.layer1 = tf.layers.dense(
                                        inputs=self.inputs,
                                        units=100,
                                        kernel_initializer=tf.initializers.random_iniform(),
                                        name="layer1")
            self.output = tf.layers.dense(
                                        inputs=self.layer1,
                                        units=self.action_size,
                                        kernel_initializer=tf.initializers.random_iniform(),
                                        name="output")

            self.Q_estimate = tf.reduce

# TODO: Network outputs
q_values = tf.layers.dense(state_in, ACTION_DIM, kernel_initializer=tf.random_uniform_initializer(0,.01), activation=tf.nn.softmax)
q_action = np.max(q_values)

# TODO: Loss/Optimizer Definition
loss = tf.square(target_in - q_action)
# loss = tf.reduce_sum(sqe, axis=1)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=.005).minimize(loss)

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action


# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= epsilon / EPSILON_DECAY_STEPS
    
    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        (next_state, reward, done, _) = env.step(np.argmax(action))

        nextstate_q_values = q_values.eval(feed_dict={
            state_in: [next_state]
        })

        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated

        target = reward + GAMMA*np.max(nextstate_q_values) 

        # Do one training step
        session.run([optimizer], feed_dict={
            target_in: [target],
            action_in: [action],
            state_in: [state]
        })

        # Update
        state = next_state
        if done:
            break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                # env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
