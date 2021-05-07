import numpy as np
import random
from IPython.display import clear_output
from collections import deque
import progressbar

import gym

from tensorflow.keras import Model,Sequential
from tensorflow.keras.layers import Dense,Embedding,Reshape
from tensorflow.keras.optimizers import Adam

environment = gym.make("Taxi-v3").env
environment.render()

print(environment.observation_space.n)
print(environment.action_space.n)

class Agent:
    def __init__(self,environment,optimizer):


        self._state_size= environment.observation_space.n
        self._action_size= environment.action_space.n
        self._optimizer = optimizer

        self.experience_replay = deque(maxlen=200)

        self.gamma = 0.6
        self.epsilon = 0.1

        self.q_network = self.build_compile_model()
        self.target_network = self.build_compile_model()
        self.align_target_model()






    def store(self,state,action,reward,next_state,terminated):
        self.experience_replay.append((state,action,reward,next_state,terminated))


    def build_compile_model(self):
        model=Sequential()
        model.add(Embedding(self._state_size,10,input_length=1))
        model.add(Reshape((10,)))
        model.add(Dense(50,activation="relu"))
        model.add(Dense(50,activation="relu"))
        model.add(Dense(self._action_size,activation="linear"))

        model.compile(loss="mse",optimizer=self._optimizer)
        return model


    def align_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())


    def act(self,state):

        if np.random.rand()<=self.epsilon:
            return environment.action_space.sample()


        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def retrain(self,batch_size):
        minibatch=random.sample(self.experience_replay,batch_size)

        for state,action,reward,next_state,terminated in minibatch:
            target = self.q_network.predict(state)

            if terminated:
                target[0][action]=reward

            else:
                t= self.target_network.predict(next_state)
                target[0][action]=reward+self.gamma*np.amax(t)

            self.q_network.fit(state,target,epochs=1,verbose=0)



optimizer = Adam(lr=0.01)
agent = Agent(environment,optimizer)

batch_size =32
num_of_episodes = 100
timesteps_per_episode =1000
print(agent.q_network.summary())


for e in range(0,num_of_episodes):

    state = environment.reset()
    state = np.reshape(state,[1,1])


    reward =0
    terminated = False


    bar = progressbar.ProgressBar(maxval=timesteps_per_episode/10, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()


    for timesteps in range(timesteps_per_episode):

        print(timesteps)

        action = agent.act(state)


        next_state,reward,terminated,info = environment.step(action)
        next_state = np.reshape(next_state,[1,1])
        agent.store(state,action,reward,next_state,terminated)

        state = next_state

        if terminated:
            agent.align_target_model()
            break

        if len(agent.experience_replay)>batch_size:
            agent.retrain(batch_size)


        if timesteps%10==0:
            bar.update(timesteps/10+1)

    bar.finish()

    if (e + 1) % 10 == 0:
        print("**********************************")
        print("Episode: {}".format(e + 1))
        environment.render()
        print("**********************************")

