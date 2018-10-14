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
import random
import time
import unicycle as ucl
from tensorflow import keras

from SumTree import SumTree

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # Allocated memory of GPU grow

from tensorflow.keras.backend import set_session

set_session(tf.Session(config=config))

np.seterr(divide='ignore')
np.set_printoptions(precision=2, suppress=True)

# basic prmts
BATCH_SIZE = 256
MAX_SAMPLES = 100000

GAMMA = 0.95

MAX_EPSILON = 1
MIN_EPSILON = 0.001
LAMBDA = 0.0001

learning_rate = 1e-4


class Environment:
    def __init__(self, render=False, **kwargs):
        self.env = ucl.Unicycle(render=render, **kwargs)

    def replay(self, policy):
        step = 0
        time_step = 1 / 30
        state = self.env.reset()

        while True:
            s = time.time()
            action = policy(state, e_greedy=False)
            state, reward, done, _ = self.env.step(action)

            if step > 2000:  # Good enough. Let's move on
                break
            if done:
                break
            step += 1

            e = time.time()

            time.sleep(max(time_step - (e - s), 0.0001))
        return step

    def close(self):
        p.resetSimulation()
        p.disconnect()

    def record(self, dir):
        p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, dir)


class Agent:
    def __init__(self, s_size=19):
        self.main = Network(input_size=s_size)
        self.target = Network(input_size=s_size)
        self.target.model.set_weights(self.main.model.get_weights())  # copy weights from main to target

        self.memory = Memory(max_sample=MAX_SAMPLES)
        self.steps = 0

    def policy(self, state, e_greedy=True):
        # e-greedy policy
        eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-LAMBDA * self.steps)
        if (np.random.random_sample() < eps) and e_greedy:
            return np.random.randint(1, self.main.a_size)
        else:
            return np.argmax(self.main.model.predict(state.reshape(1, self.main.s_size)))

    def random_policy(self, _):
        return random.randint(0, 3)

    def displacement(self, samples):
        n_batch = len(samples)
        states, actions, rewards, states_ = np.transpose(samples)

        q = self.main.predict(np.vstack(states))
        q_ = self.target.predict(np.vstack(states_))

        q_old = np.copy(q)

        for i in range(n_batch):
            if rewards[i] == 0:
                q[i][actions[i]] = rewards[i]
            else:
                q[i][actions[i]] = rewards[i] + GAMMA * q_[i][self.policy(states_[i])]

        error = [q[i][actions[i]] - q_old[i][actions[i]] for i in range(n_batch)]

        alpha = 1.
        epsilon = 0.001

        priors = (np.abs(error) + epsilon) ** alpha
        return priors, q, states

    def learn(self, environment):
        step = self.get_samples(environment, self.policy)  # play a game until done

        indices = []
        samples = []
        for idx, sample in self.memory.sample(BATCH_SIZE):
            indices.append(idx)
            samples.append(sample)

        priors, q, states = self.displacement(samples)

        self.main.train(np.vstack(states), np.vstack(q), epochs=1)

        if self.steps % 100 == 0:
            self.target.model.set_weights(self.main.model.get_weights())
        self.steps += 1

        for idx, prior in zip(indices, priors):
            self.memory.tree.update(idx, prior)

        return step

    def get_samples(self, environment, policy):
        step = 0
        state = environment.reset()
        samples = []

        while True:
            action = policy(state)
            state_, reward, done, [] = environment.step(action)

            samples.append([state, action, reward, state_])

            state = state_

            if done:
                break
            step += 1

        priors, q, _ = self.displacement(samples)

        for prior, sample in zip(priors, samples):
            self.memory.add(prior, sample)

        return step


class Network:
    def __init__(self, input_size=19, output_size=4):
        self.s_size = input_size
        self.a_size = output_size
        self.model = self._build_network()

    def _build_network(self):
        model = keras.Sequential([
            keras.layers.Dense(256, activation='relu',
                               input_shape=(self.s_size,),
                               name='layer_1',
                               kernel_initializer='glorot_uniform',
                               bias_initializer='glorot_uniform'
                               ),
            keras.layers.Dense(256, activation='relu',
                               name='layer_2',
                               kernel_initializer='glorot_uniform',
                               bias_initializer='glorot_uniform'
                               ),
            keras.layers.Dense(self.a_size, activation='linear')
        ])

        opt = keras.optimizers.Adam(lr=learning_rate)
        model.compile(loss=tf.losses.huber_loss, optimizer=opt)

        return model

    def predict(self, s):
        return self.model.predict(s)

    def train(self, x, y, epochs=1):
        self.model.fit(x, y, batch_size=BATCH_SIZE, epochs=epochs, verbose=False)


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
            p = np.random.random() * total

            batch.append(self.tree.get_data(p))

        return batch


def main(get_image=False, weights_save=None):
    env = Environment(sigma=0.02, down=1.2, get_image=get_image)
    s_size = env.env.s_size

    agent = Agent(s_size=s_size)
    agent.main.model.load_weights("data/" + weights_save + "_main.h5", by_name=True)
    agent.target.model.load_weights("data/" + weights_save + "_main.h5", by_name=True)

    max_step = 0
    step_que = []
    epoch_cnt = 0
    start = time.time()

    try:
        agent.get_samples(env.env, agent.random_policy)
        while True:
            step_cnt = agent.learn(env.env)
            step_que.append(step_cnt)
            if len(step_que) > 10:
                step_que.pop(0)
            epoch_cnt += 1
            if epoch_cnt % 1000 == 0:
                print("mean of step_que = {:.1f}".format(np.mean(step_que)) +
                      "    var of step_que = {:.1f}".format(np.std(step_que)) +
                      "    time taken = {:.1f}".format(time.time() - start) +
                      "    current epoch = {}".format(epoch_cnt)
                      )
            if min(step_que) > 1000:
                print(step_que)
                break
    finally:
        env.close()
        print("model saved")
        end = time.time()
        print("total time = {},    epoch = {},".format(end - start, epoch_cnt) +
              "    mean of recent steps = {}".format(np.mean(step_que))
              )
        env0 = Environment(render=True, sigma=0.02, down=1.0, get_image=get_image)
        env0.replay(agent.policy)
        if weights_save:
            agent.main.model.save_weights("data/" + weights_save + "_main.h5")
            agent.target.model.save_weights("data/" + weights_save + "_target.h5")
        env0.close()


if __name__ == '__main__':
    with tf.device('/cpu'):
        main(weights_save="unicycle_DDQN", get_image=False)
        # main(get_image=True)
