#!/usr/bin/env python
# coding: utf-8

# ## DQN module for unicycle learning
# 
# input with 19 coord data
# output with 4 torque for one-hot
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
        
    def get_samples(self, agent):
        step = 0
        state = self.env.reset()

        while True:
            action = agent.policy(state)
            state_, reward, done, [] = self.env.step(action)

            agent.observe([state, action, reward, state_])
            
            state = state_
            
            if done:
                break
            step += 1
        return step
                
    def replay(self, policy):
        step = 0
        time_step = 1/30
        state = self.env.reset()

        while True:
            s = time.time()
            action = policy(state, e_greedy=False)
            state, reward, done, _ = self.env.step(action)

            if step > 10000:  # Good enough. Let's move on
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
        self.network = Network(input_size=s_size)
        self.memory = Memory(max_sample=MAX_SAMPLES)
        self.steps = 0
        
    def policy(self, state, e_greedy=True):
        eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-LAMBDA * self.steps)
        if (np.random.random_sample() < eps) and e_greedy:
            return np.random.randint(1, self.network.a_size)
        else:
            return np.argmax(self.network.predict(state.reshape(1, self.network.s_size)))
        
    def observe(self, sample):
        self.memory.add(sample)
        
    def learn(self, environment, dis=GAMMA):
        step = environment.get_samples(self)
        batch = self.memory.sample(BATCH_SIZE)
        n_batch = len(batch)
        
        states, actions, rewards, states_ = np.transpose(batch)

        q = self.network.predict(np.vstack(states))
        q_ = self.network.predict(np.vstack(states_))
        
        for i in range(n_batch):
            if rewards[i] == -1:
                q[i][actions[i]] = rewards[i]
            else:
                q[i][actions[i]] = rewards[i] + dis * max(q_[i])
        
        self.network.train(np.vstack(states), np.vstack(q), epochs=1)
        self.steps += 1

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
        model.compile(loss='mse', optimizer=opt)
        
        return model

    def predict(self, s):
        return self.model.predict(s)
        
    def train(self, x, y, epochs=1):
        self.model.fit(x, y, batch_size=BATCH_SIZE, epochs=epochs, verbose=False)


class Memory:
    def __init__(self, max_sample=MAX_SAMPLES):
        self.max_sample = max_sample
        self.samples = []
        
    def add(self, sample):
        self.samples.append(sample)
        
        if len(self.samples) > self.max_sample:
            self.samples.pop(0)
            
    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)


def main(weights_save=None, get_image=False):
    env = Environment(sigma=0.02, down=1.2, get_image=get_image)
    s_size = env.env.s_size

    agent = Agent(s_size=s_size)
    #agent.network.model.load_weights("data/unicycle_DQN.h5", by_name=True)

    max_step = 0
    step_que = []
    epoch_cnt = 0
    start = time.time()

    try:
        while True:
            step_cnt = agent.learn(env)
            step_que.append(step_cnt)
            if len(step_que) > 10:
                step_que.pop(0)
            epoch_cnt += 1
            if np.mean(step_que) > max_step:
                max_step = np.mean(step_que)
                print("mean steps of que = {:.2f}".format(max_step) +
                      "    minimum steps in que = {}".format(min(step_que)) +
                      "    time taken = {:.2f}".format(time.time() - start) +
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
            agent.network.model.save_weights("data/" + weights_save + ".h5")
        env0.close()


if __name__ == '__main__':
    with tf.device('/cpu'):
        main(weights_save="unicycle_DQN", get_image=False)
        # main(weights_save="unicycle_DQN", get_image=True)
