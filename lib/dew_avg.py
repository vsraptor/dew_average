import numpy as np
import matplotlib.pylab as plt
from stats import stats
from qlearn import *


class DEWAvg :

	def __init__(self, nstates=10,vmin=0, vmax=100, zero_freqs=True, learn_rate=0.5, gamma=0.9):
		self.nstates = nstates
		self.vmin = vmin
		self.vmax = vmax
		self.vrange = self.vmax - self.vmin

		self.zero_freqs = zero_freqs
		self.last_pred = 0

		#Reinforcment learning for multi-step prediction
		self.ql = QLearn(nstates, learn_rate, gamma)
		self.last_state = 0

		#we collect the whole signal for plotting&stats purposes,
		# .. but just ctx_len buffer will be enough for functioning model
		self.orig = [0]
		self.sig = [0]
		self.nope = [ ] #collect assumed/copied but not predicted : see. zero_freqs
		self.yhat = [ ]

		self.cnt = 0

	def learn(self, data):
		#cut values to min-max range
		if data <= self.vmin : data = self.vmin
		if data >= self.vmax : data = self.vmax

		#smoothing : limit the number of states, so we can do predictions, otherwize we need fuzzy calculations
		state = self.encode(data)
		self.ql.update(self.last_state, state, reward=1)
		self.last_state = state

		self.cnt += 1
		return state

	def predict(self, data):
		state = self.encode(data)
		rv = self.ql.predict(state)

		#gives better prediction, corrects for min/max state spikes
		if self.zero_freqs :
			if rv == 0 or rv == self.nstates :
				rv = self.last_pred
				self.nope.append(rv)
			else :
				rv = self.decode(rv)
				self.nope.append(0)

			self.last_pred = self.sig[-1] #or we can use rv, which is the last predicted
			return rv
		else :
			self.nope.append(0)
			return self.decode(rv)


	def train(self, data, log=True):
		learned = self.learn(data)
		pred = self.predict(data)

		#collect signal and prediction so we can calculate stats
		self.sig.append(self.decode( learned ))
		self.yhat.append(float(pred))
		if log and len(self.yhat) > 0 : print "%s : %s => %s : %s" % (self.cnt, self.sig[-1] , self.yhat[-1], self.sig[-1] - self.yhat[-1])


	def batch_train(self,sig, log=False):
		self.orig = np.concatenate([[0], sig]) #keep original signal around
		for s in sig : self.train(s, log)


	def encode(self, values):
		rv = np.floor(self.nstates * ((values - self.vmin)/float(self.vrange)) )
		if isinstance(rv, np.ndarray) : return rv.astype(np.uint16)
		return int(rv)

	def decode(self, values):
		rv = np.floor( (values * self.vrange) / float(self.nstates) ) + self.vmin
		return rv





	def plot(self, skip=0, nope=False, original=True):
		fig = plt.figure()
		ax = fig.add_subplot(111)

		if original : plt.plot(self.orig[skip:], color='yellow')
		plt.plot(self.sig[skip:], color='blue')
		plt.plot(self.yhat[skip:], color='green')

		#plot non predicted
		if nope and self.zero_freqs :
	 		nope = np.array(self.nope, dtype=np.int)[skip:]
			nidxs = (np.where(nope > 0))[0]
			plt.plot(nidxs, nope[nidxs], 'r.')

		metrics = self.stats(skip=skip, original=original)
		fig.suptitle("s:%s, min:%s, max:%s" % (self.nstates, self.vmin, self.vmax ))
		y = 0.998
		for m in ['mape','nll', 'mae', 'rmse', 'r2'] :
			metric = "%s: %s" % (str(m).upper(), metrics[m])
			plt.text(0.998,y, metric, horizontalalignment='right',  verticalalignment='top', transform=ax.transAxes)
			y -= 0.03

		plt.grid()
		plt.tight_layout()


	def stats(self, skip=0, original=False):
		data = None
		if original :
			data = np.array(self.orig)[skip:]
		else :
			data = np.array(self.sig)[skip:]

		yhat = np.array(self.yhat + [0])[skip:]

		m = {}
		m['mape'] = "%.3f%%" % (stats.mape(data,yhat) * 100)
		m['mae'] =  "%.3f" % stats.mae(data,yhat)
		m['rmse'] = "%.3f" % stats.rmse(data,yhat)
		m['r2'] = "%.3f%%" % (stats.r2(data,yhat) * 100)
		m['nll'] = "%.3f" % stats.nll(data,yhat)

		print "==== MODEL stats =============="
		print "mape: %s " % m['mape']
		print "mae : %s" % m['mae']
		print "rmse: %s" % m["rmse"]
		print "r2: %s" % m['r2']
		print "nll: ", m['nll']
		print "resolution: %s" % (self.vrange/float(self.nstates))

		return m




