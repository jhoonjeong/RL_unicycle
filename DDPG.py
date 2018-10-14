#!/usr/bin/env python
# coding: utf-8

# ## DDQN module for unicycle learning
#
# input with 19 coord data
# ouput with 4 torque for one-hot
# https://www.tensorflow.org/guide/keras
# https://keras.io/getting-started/functional-api-guide/

# https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/


import numpy as np
import tensorflow as tf
import pybullet as p
import time
import unicycle as ucl
from SumTree import SumTree

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # Allocated memory of GPU grow

np.seterr(divide='ignore')
np.set_printoptions(precision=2, suppress=True)

from tensorflow.keras.backend import set_session

set_session(tf.Session(config=config))

# basic prmts
BATCH_SIZE = 64
MAX_SAMPLES = 1000000

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.001
LAMBDA = 0.01

a_learning_rate = 1e-4
c_learning_rate = 1e-3

TAU = 0.001


class Environment:
    def __init__(self, render=False, **kwargs):
        self.env = ucl.Unicycle(render=render, continuous=True, **kwargs)

    def get_samples(self, actor):
        step = 0
        state = self.env.reset()

        while True:
            action = actor.policy(state)
            state_, reward, done, [] = self.env.step(action)

            actor.observe([state, action, reward, state_, done])

            state = state_

            if done:
                break
            step += 1
        return step

    def replay(self, policy):
        step = 0
        time_step = 1 / 30
        state = self.env.reset()

        while True:
            s = time.time()
            action = policy(state.reshape([1, len(state)]))
            state, reward, done, _ = self.env.step(action)

            if step > 1000:  # Good enough. Let's move on
                break
            if done:
                break
            step += 1

            e = time.time()

            time.sleep(max(time_step - (e - s), 0.00001))
        return step

    def close(self):
        p.resetSimulation()
        p.disconnect()

    def record(self, dir):
        p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, dir)


class ActorNetwork:
    def __init__(self, session, a_bound, s_size, a_size, name=None):
        self.session = session
        self.a_bound = a_bound
        self.s_size = s_size
        self.a_size = a_size
        self.name = name

        self._build_network()

    def _build_network(self):
        name = self.name
        with tf.variable_scope(name):
            self.S = tf.placeholder(tf.float32, shape=[None, self.s_size], name=name + "S")

            self.W1 = tf.get_variable(name + "W1", shape=[self.s_size, 400],
                                      initializer=tf.glorot_uniform_initializer())
            self.b1 = tf.get_variable(name + "b1", shape=[400],
                                      initializer=tf.glorot_uniform_initializer())
            self.layer1 = tf.nn.relu(tf.matmul(self.S, self.W1) + self.b1)

            self.W2 = tf.get_variable(name + "W2", shape=[400, 300],
                                      initializer=tf.glorot_uniform_initializer())
            self.b2 = tf.get_variable(name + "b2", shape=[300],
                                      initializer=tf.glorot_uniform_initializer())
            self.layer2 = tf.nn.relu(tf.matmul(self.layer1, self.W2) + self.b2)

            self.W3 = tf.get_variable(name + "W3", shape=[300, self.a_size],
                                      initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            self.b3 = tf.get_variable(name + "b3", shape=[self.a_size],
                                      initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

            self._mu_pred = tf.multiply(tf.nn.tanh(tf.matmul(self.layer2, self.W3) + self.b3), self.a_bound)

            self.Y = tf.placeholder(tf.float32, shape=[None, self.a_size], name=name + "Y")

            self.grad_Q = tf.placeholder(tf.float32, shape=[None, self.a_size], name="grad_Q")

            self._J = tf.multiply(self.grad_Q, self._mu_pred)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=a_learning_rate)
            self._grads_and_vars = self.optimizer.compute_gradients(self._J,
                                                                    var_list=tf.trainable_variables(scope=name))
            self._train = self.optimizer.apply_gradients(self._grads_and_vars)

    def predict(self, s_stack):
        return self.session.run(self._mu_pred, feed_dict={self.S: s_stack})

    def train(self, s_stack, grad_Q):
        return self.session.run(self._train, feed_dict={self.S: s_stack, self.grad_Q: grad_Q})


class Actor:
    def __init__(self, session, a_bound, s_size=19, a_size=2):
        self.session = session
        self.steps = 0
        self.input_size = s_size
        self.output_size = a_size
        self.a_bound = a_bound

        self.original = ActorNetwork(session, a_bound, self.input_size, self.output_size, name="a_original")
        self.target = ActorNetwork(session, a_bound, self.input_size, self.output_size, name="a_target")

        self.o_prmts = tf.trainable_variables(scope="a_original")
        self.t_prmts = tf.trainable_variables(scope="a_target")
        self.tau = TAU
        self.cp_network_params = [t.assign(o.value()) for t, o in zip(self.t_prmts, self.o_prmts)]

        self.update_network_params = \
            [t.assign(tf.multiply(o.value(), self.tau) + tf.multiply(t.value(), 1. - self.tau))
             for t, o in zip(self.t_prmts, self.o_prmts)]

    def policy_one(self, state):
        policy = self.policy(state.reshape((1, self.input_size)))
        return policy

    def policy(self, states):
        return self.original.predict(states)

    def policy_target_one(self, state):
        action = self.policy_target(state.reshape((1, self.input_size)))
        return action

    def policy_target(self, states):
        return self.target.predict(states)

    def target_cp(self):
        return self.session.run(self.cp_network_params)

    def target_update(self):
        return self.session.run(self.update_network_params)


class CriticNetwork:
    def __init__(self, session, s_size, a_size, name=None):
        self.session = session
        self.s_size = s_size
        self.a_size = a_size
        self.name = name
        self._build_network()

    def _build_network(self):
        name = self.name
        with tf.variable_scope(name):

            self.S = tf.placeholder(tf.float32, shape=[None, self.s_size], name=name+"S")
            self.A = tf.placeholder(tf.float32, shape=[None, self.a_size], name=name+"A")

            self.W1 = tf.get_variable(name+"W1", shape=[self.s_size, 400],
                                 initializer=tf.glorot_uniform_initializer())
            self.b1 = tf.get_variable(name+"b1", shape=[400],
                                 initializer=tf.glorot_uniform_initializer())
            self.layer1 = tf.nn.relu(tf.matmul(self.S, self.W1) + self.b1)

            self.W2 = tf.get_variable(name+"W2", shape=[400, 300],
                                 initializer=tf.glorot_uniform_initializer())
            self.W2_1 = tf.get_variable(name+"W2_1", shape=[self.a_size, 300],
                                   initializer=tf.glorot_uniform_initializer())
            self.b2 = tf.get_variable(name+"b2", shape=[300],
                                 initializer=tf.glorot_uniform_initializer())
            self.layer2 = tf.nn.relu(tf.matmul(self.layer1, self.W2) + tf.matmul(self.A, self.W2_1) + self.b2)

            self.W3 = tf.get_variable(name+"W3", shape=[300, 1],
                                 initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            self.b3 = tf.get_variable(name+"b3", shape=[1],
                                 initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

            vars = tf.trainable_variables(scope=self.name)
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'b' not in v.name ]) * 0.01

            self.Q_pred = (tf.matmul(self.layer2, self.W3) + self.b3 + lossL2)

            self.Y = tf.placeholder(tf.float32, shape=[None, 1], name=name+"Y")

            self._loss = tf.squared_difference(self.Q_pred, self.Y)

            self._train = tf.train.AdamOptimizer(learning_rate=c_learning_rate).minimize(
                tf.reduce_mean(self._loss))

            alpha = tf.constant(1., dtype=tf.float32)
            epsilon = tf.constant(0.001, dtype=tf.float32)

            self._prior = tf.pow(tf.add(self._loss, epsilon), alpha)

            self._grad_Q = tf.gradients(self.Q_pred, self.A)

    def predict(self, s_stack, a_stack):
        return self.session.run(self.Q_pred, feed_dict={self.S: s_stack, self.A: a_stack})

    def train(self, s_stack, a_stack, y_stack):
        return self.session.run([self._grad_Q, self._prior, self.Q_pred, self._train],
                                feed_dict={self.S: s_stack, self.A: a_stack, self.Y: y_stack})

    def loss(self, s_stack, a_stack, y_stack):
        return self.session.run(self._loss,
                                feed_dict={self.S: s_stack, self.A: a_stack, self.Y: y_stack})


class Critic:
    def __init__(self, session, s_size=19, a_size=2):
        self.session = session
        self.steps = 0
        self.input_size = s_size + a_size
        self.s_size = s_size
        self.a_size = a_size

        self.original = CriticNetwork(session, s_size, a_size, name="c_original")
        self.target = CriticNetwork(session, s_size, a_size, name="c_target")

        self.o_prmts = tf.trainable_variables(scope="c_original")
        self.t_prmts = tf.trainable_variables(scope="c_target")

        self.cp_network_params = [t.assign(o.value()) for t, o in zip(self.t_prmts, self.o_prmts)]
        self.update_network_params = \
            [t.assign(tf.multiply(o.value(), TAU) + tf.multiply(t.value(), 1. - TAU))
             for t, o in zip(self.t_prmts, self.o_prmts)]

    def value(self, states_, actions_, rewards, dones):
        q_ = self.target.predict(states_, actions_)
        y = np.vstack(rewards) + np.logical_not(np.vstack(dones)) * GAMMA * q_
        return y

    def value_one(self, state_, action_, reward, done):
        q_ = self.target.predict(state_.reshape((1, self.s_size)), action_)
        y = np.reshape(reward, (1, )) + np.logical_not(done) * GAMMA * q_
        return np.reshape(y, (1,))

    def differences(self, samples, y_stack):
        states, actions, rewards, dones, states_ = samples
        alpha, epsilon = 0., 0.001
        prior = (self.original.loss(states, actions, y_stack) + epsilon) ** alpha
        return prior, states, actions

    def differences_one(self, sample, y):
        state, action, reward, done, state_ = sample
        samples = (state.reshape((1, self.s_size)),
                   action.reshape((1, self.a_size)),
                   np.reshape(reward, (1, 1)),
                   np.reshape(done, (1, 1)),
                   state_.reshape((1, self.s_size))
                   )
        return self.differences(samples, y.reshape((1, 1)))

    def target_cp(self):
        return self.session.run(self.cp_network_params)

    def target_update(self):
        return self.session.run(self.update_network_params)


class Memory:
    def __init__(self, max_sample=MAX_SAMPLES):
        self.max_sample = max_sample
        self.tree = SumTree(self.max_sample)

    def add(self, prior, sample):
        self.tree.add(prior, sample)

    def sample(self, n):
        batch = []
        total, _ = self.tree.total()

        for i in range(n):
            p = np.float64(np.random.random() * total)
            idx, data = self.tree.get_data(p)
            batch.append([idx, data])

        return batch


def DDPG(env, actor, critic):
    memory = Memory(max_sample=MAX_SAMPLES)
    step_que = []
    start = time.time()
    epoch_cnt = 0
    s_size = env.env.s_size
    a_size = 2

    while True:
        state = env.env.reset().reshape((s_size))
        done = False

        theta = 0.15
        sigma = 0.02
        time_step = 1/30

        step = 0

        OU_noise = np.zeros(a_size)

        while not done:
            OU_noise += -theta * OU_noise * time_step
            OU_noise += sigma * np.sqrt(time_step) * np.random.randn(a_size)

            action = actor.policy_one(state)
            action += 2. * OU_noise

            state_, reward, done, _ = env.env.step(action)
            state_ = state_.reshape((s_size))

            sample = (state, action, reward, done, state_)

            action__ = actor.policy_target_one(state_)

            y_stack = critic.value_one(state_, action__, reward, done)

            prior, _, _ = critic.differences_one(sample, y_stack)

            memory.add(np.float64(prior), sample)

            state = state_

            step += 1

            if sum(step_que) + step > BATCH_SIZE:
                indices = []
                states = []
                actions = []
                rewards = []
                dones = []
                states_ = []

                for idx, sample in memory.sample(BATCH_SIZE):
                    indices.append(idx)
                    states.append(sample[0])
                    actions.append(sample[1])
                    rewards.append(sample[2])
                    dones.append(sample[3])
                    states_.append(sample[4])

                actions_ = actor.policy_target(np.vstack(states_))

                y_stack = critic.value(states_, actions_, rewards, dones)

                grad_var, priors, Q_pred, _ = critic.original.train(states,
                                                                    np.reshape(actions, [BATCH_SIZE, a_size]),
                                                                    y_stack)

                actor.original.train(states, - grad_var[0] / BATCH_SIZE)
                actor.target_update()
                critic.target_update()
        step_que.append(step)
        if sum(step_que) > BATCH_SIZE:
            print("time taken ={:7.1f}".format(time.time() - start) +
                  "   current epoch ={:5d}".format(epoch_cnt) +
                  "   mean of Q_pred = {:6.2f}".format(np.mean(Q_pred)) +
                  "   step = {}".format(step)
                  )
        if min(step_que[-5:]) > 1000:
            break
        epoch_cnt += 1


# def variable_noise(o_name, t_name):
#     theta_o = tf.get_collection(tf.trainable_variables, scope=o_name)
#     theta_t = tf.get_collection(tf.trainable_variables, scope=t_name)
#     for o_, t_ in zip(theta_o, theta_t):
#         t_.assign(tau * o_.value + (1 - tau) * t_.value)


def main(session, get_image=False, weights_save=None):
    #env = Environment(sigma=0.01, down=1.0, get_image=get_image)
    env = Environment(sigma=0.02, down=1.2, get_image=get_image)
    s_size = env.env.s_size
    a_size = 2
    a_bound = 1.

    actor = Actor(session, a_bound, s_size=s_size, a_size=a_size)
    critic = Critic(session, s_size=s_size, a_size=a_size)

    writer = tf.summary.FileWriter('data/DDPG')

    session.run(tf.global_variables_initializer())

    actor.target_cp()
    critic.target_cp()

    try:
        DDPG(env, actor, critic)
    finally:
        env.close()

        if weights_save:
            saver = tf.train.Saver()
            save_path = saver.save(session, "data/tf/" + weights_save + ".ckpt")
            print("model saved")

        env = Environment(render=True, sigma=0.02, down=1.0, get_image=False)
        #env.render()
        env.replay(actor.policy_one)
        env.close()


if __name__ == '__main__':

    with tf.device('/cpu'):
        with tf.Session() as sess:

            main(sess, get_image=False, weights_save="DDPG")
        # main(get_image=True)