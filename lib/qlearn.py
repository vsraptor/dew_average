import numpy as np
import matplotlib.pylab as plt

class QLearn:

#Q(s,a) = reward + gamma * max(Q(s', all))
#Q(s,a) += lrate * ( reward(s,a,s') + g * max(Q(s')) - Q(s,a) )
#SARSA: Q(s,a) += lrate * ( reward(s,a,s') + g * Q(s',a) - Q(s,a) )


	def __init__(self, nstates=3, learn_rate=0.5, gamma=0.9) :
		self.nstates = nstates
		self.lrate = learn_rate
		self.gamma = gamma

		self.qmap = np.zeros((nstates, nstates))

	def update(self, s0, s1, reward, log=False) :
		delta = reward + ( self.gamma * self.qmap[s1,:].max() ) - self.qmap[s0,s1]
		self.qmap[s0,s1] += self.lrate * delta
		if log: print s0, s1, reward, delta, self.qmap[s0,s1]

	def predict(self, state):
		return self.qmap[state,:].argmax()

	def act(self, state):
		if np.random.random() > self.epsilon :
			return self.qmap[state,:].argmax()
		else:
			return np.random.randint(0,self.nstates)


	def show_map(self,data=None):
		if data is None : data = self.qmap
		plt.figure()
		plt.imshow(data, cmap='Greys', interpolation='nearest', aspect='auto')
