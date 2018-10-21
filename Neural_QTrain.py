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
GAMMA =  .85# discount factor
INITIAL_EPSILON = 1# starting value of epsilon
FINAL_EPSILON = .05 # final value of epsilon
EPSILON_DECAY_STEPS = 1000# decay period
HIDDEN_UNITS = 128
MEMORYSIZE = 1000
BATCH_SIZE = 500
batch_size = BATCH_SIZE

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
with tf.variable_scope("DQN"):
    #Network architecture
   
    q_values = tf.layers.dense(
                                inputs=state_in,
                                units=20,
                                kernel_initializer=tf.initializers.random_uniform(),
                                activation=tf.nn.relu,
                                name="layer1")

    q_values = tf.layers.dense(
                                inputs=q_values,
                                units=ACTION_DIM,
                                kernel_initializer=tf.initializers.random_uniform(),
                                # activation=tf.nn.relu,
                                name="layerq")
    # q_values = outputs[:,-1,:]
    '''q_values is a vector of the q_values for each action given the current state. e.g. [0.1, -0.6]
    q_action is just the q_value for the specific action taken. Mathematically, actions will be a one_hot vector like [0, 1], so the multiplication of actions with values will be:
    q_action = [0.1, -0.6] * [0, 1] = [0, -0.6]
    tf.reduce_sum() has the function of compiling this down to the Q(s,a)(q value), for that specific state because it is currently like this:
    [Q(s, action 1), Q(s, action 2)]
    Therefore, q_action becomes: -0.6
    '''
    q_action = tf.reduce_sum(tf.multiply(q_values, action_in), axis=1)
    loss = tf.reduce_mean(tf.square(target_in - q_action))
    # l_loss = tf.reduce_mean(tf.square(target_in - q_action))
    optimiser = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

with tf.variable_scope("DDQN"):
    #Network architecture
   
    dq_values = tf.layers.dense(
                                inputs=state_in,
                                units=100,
                                kernel_initializer=tf.initializers.random_uniform(),
                                activation=tf.nn.relu,
                                name="layer1")
    
    dq_values = tf.layers.dense(
                                inputs=dq_values,
                                units=ACTION_DIM,
                                kernel_initializer=tf.initializers.random_uniform(),
                                # activation=tf.nn.relu,
                                name="layerq")

    '''q_values is a vector of the q_values for each action given the current state. e.g. [0.1, -0.6]
    q_action is just the q_value for the specific action taken. Mathematically, actions will be a one_hot vector like [0, 1], so the multiplication of actions with values will be:
    q_action = [0.1, -0.6] * [0, 1] = [0, -0.6]
    tf.reduce_sum() has the function of compiling this down to the Q(s,a)(q value), for that specific state because it is currently like this:
    [Q(s, action 1), Q(s, action 2)]
    Therefore, q_action becomes: -0.6
    '''
    
    dloss = tf.reduce_mean(tf.square(q_values - dq_values))
    # l_loss = tf.reduce_mean(tf.square(target_in - q_action))
    doptimiser = tf.train.AdamOptimizer(learning_rate=0.001).minimize(dloss)



class BatchMemory:
    def __init__(self, memory):
        self.memory = []
        self.max_memory = memory
        self.update_runner = 0

    def add(self, elem):
        if self.size > self.max_memory:
            self.memory[self.update_runner] = elem
        else:
            self.memory.append(elem)

    def get_single(self):
        return self.memory[self.rand_access]  

    def get_batch(self, batch_size):
        if batch_size > self.size:
            return random.sample(self.memory, self.size)
        return random.sample(self.memory, batch_size-1)

    @property
    def size(self):
        return len(self.memory)

    @property
    def rand_access(self):
        return random.randint(0,self.size) if self.size == 0 else random.randint(0,self.size-1)

# TODO: Network outputs


# TODO: Loss/Optimizer Definition

memory = BatchMemory(MEMORYSIZE)
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

ave_reward = 0
# Main learning loop
next_s_np = []
reward_np = []
done_np = []
action_np = []
state_np = []

# frame = []
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= epsilon / EPSILON_DECAY_STEPS
    
    # Move through env according to e-greedy policy
    for step in range(STEP):

        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))
        nextstate_q_values = dq_values.eval(feed_dict={
            state_in: [next_state]
        })
        
        next_s_np.append(next_state)
        reward_np.append(reward)
        done_np.append(done)
        action_np.append(action)
        state_np.append(state)
        record = [np.array(entry) for entry in [state, action, reward, next_state, done]]

        memory.add(record)
        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated
        

        # Update
        state = next_state
        if done:
            break

        batch = np.stack(memory.get_batch(batch_size), axis=1)
        state_b =  np.vstack(batch[0])
        action_b = np.vstack(batch[1]) 
        reward_b = np.vstack(batch[2]) 
        next_state_b = np.vstack(batch[3])
        finish_b = np.vstack(batch[4]) 
        feed = {state_in: next_state_b}
        q_next_state = session.run(dq_values, feed)
        target_q = []
        for k in range(0,len(batch[0])):
            
            # print('preloop')

            if finish_b[k]:
                target_q.append(*reward_b[k])
            else:
                temp = reward_b[k] + GAMMA*np.max(q_next_state[k])
                target_q.append(*temp)

        target_q = np.array(target_q)
        session.run([optimiser], feed_dict={
            target_in: target_q,
            action_in: action_b,
            state_in: state_b
        })
    
    batch = np.stack(memory.get_batch(batch_size), axis=1)
    state_b =  np.vstack(batch[0,:])
    action_b = np.vstack(batch[1]) 
    reward_b = np.vstack(batch[2]) 
    next_state_b = np.vstack(batch[3])
    finish_b = np.vstack(batch[4]) 
    feed = {state_in: next_state_b}
    q_next_state = session.run(q_values, feed)
    target_q = []
    for k in range(0,len(batch[0])):
        
        # print('preloop')

        if finish_b[k]:
            target_q.append(*reward_b[k])
        else:
            temp = reward_b[k] + GAMMA*np.max(q_next_state[k])
            target_q.append(*temp)

    target_q = np.array(target_q)
    session.run([doptimiser], feed_dict={
        target_in: target_q,
        action_in: action_b,
        state_in: state_b
    })
    # Test and view sample runs - can disable render to `~save time
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

