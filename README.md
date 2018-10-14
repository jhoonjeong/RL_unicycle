
Unicycle
==========



## unicycle learning ######

This is code to study reinforcement learning.

DDQN and A3C codes are based on 
[the codes of Jaromir](https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/)
Also you can find a great explaination of Jaromir on [_here_](https://jaromiru.com/)  
Steeve Huang's medium [page](https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-i-q-learning-sarsa-dqn-ddpg-72a5e0cb6287)  
Unicycle simulation is using [_pybullet_](https://github.com/bulletphysics/bullet3)
physics modeling.  
[Thomas Simonini](https://medium.freecodecamp.org/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f)
---

Should be changed list:

  * visual input and camera control
  * pybullet torque bug
  * New algorithms (PPO)
  * torque, angular acceleration reading
  * multi thread
  [link](https://stackoverflow.com/questions/2846653/how-to-use-threading-in-python)
  * some issue in tensorflow with keras. [link](https://github.com/tensorflow/tensorflow/issues/8652)
  * Add Prioritized Experience Replay
  * Native Tensorflow

Algorithms
  1. (D)DQN
  2. DDPG
  
  planning
  3. PPO
  4. TRPO
  5. A3C

> string.jeong@gmail.com

  * static case   
  gamma = 0.95  
  reward (0, -1) punishment  
  min.step_que = 1000  
  sigma = 0.02  
  down = 1.2  
  
  * vision input  
  160 * 120 grayscale  
  takes more than time_step  
  make it systematic  
  make network input size
  
