# -*- coding: utf-8 -*-

import numpy as np
import time, cPickle
from scipy import optimize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class spLSI:
	# implementation of probablistic latent semantic indexing

	def __init__(self):
		# do nothing particularly
		pass

	def setData(self,data,label):
		# data is required to be given in a two dimensional numpy array (nDocuments,nVocabulary)
		# with each element representing the number of times observed
		# label is required to be given in an one dimensional numpy array (nDocuments,)

		# set parameters
		self.data = data
		self.label = label
		self.nDocuments = data.shape[0]
		self.nVocabulary = data.shape[1]
		
	def sumMulti(self,doc,qZ,eta,nd):
		return np.prod(np.power(np.dot(qZ,np.exp(eta/nd)),doc[:]))

	def sumMultiOmit(self,doc,qZ,eta,nd):
		return np.full(qZ.shape[0],self.sumMulti(doc,qZ,eta,nd))/(np.dot(qZ,np.exp(eta/nd)))

	def costFunction(self,eta,*args):
		# return cost function to minimize
		lEta = 0.0
		data,qZAll,zetaAll,labelAll = args

		# iterate over all documents
		for d in range(data.shape[0]):
			doc = data[d,:]
			qZ = qZAll[d,:,:]
			zeta = zetaAll[d]
			label = labelAll[d]
			nd = float(doc.sum())
			lEta += label/nd*np.dot(np.dot(qZ,eta),doc) - 1.0/zeta*self.sumMulti(doc,qZ,eta,nd)
		return -lEta

	def gradient(self,eta,*args):
		# return gradient for cost function
		data,qZAll,zetaAll,labelAll = args
		lEtaGradient = np.zeros(eta.shape[0])

		# iterate over all documents
		for d in range(data.shape[0]):
			doc = data[d,:]
			qZ = qZAll[d,:,:]
			zeta = zetaAll[d]
			label = labelAll[d]
			nd = float(doc.sum())
			buff = label*(qZ*doc.reshape((qZ.shape[0],1))).sum(axis=0)
			buff += -1.0/zeta*((qZ*np.exp(eta/nd).reshape(1,qZ.shape[1])*doc.reshape((qZ.shape[0],1)))*self.sumMultiOmit(doc,qZ,eta,nd).reshape((qZ.shape[0],1))).sum(axis=0)
			lEtaGradient += buff/nd
		return -lEtaGradient

	def solve(self,nTopics=10,epsilon=1e-6):

		# set additional parameters
		self.nTopics = nTopics
		self.epsilon = epsilon

		# define p(z),p(d|z),p(w|z)
		self.pZ = np.zeros(self.nTopics,dtype=np.float64)
		self.pD = np.zeros((self.nTopics,self.nDocuments),dtype=np.float64)
		self.pW = np.zeros((self.nTopics,self.nVocabulary),dtype=np.float64)

		# define eta and zeta
		self.eta = np.full(self.nTopics,1.0,dtype=np.float64)
		self.zeta = np.full(self.nDocuments,1.0,dtype=np.float64)

		# define and initialize qZ
		self.qZ = np.random.rand(self.nDocuments,self.nVocabulary,self.nTopics)
		self.qZ /= self.qZ.sum(axis=2).reshape((self.nDocuments,self.nVocabulary,1))
		
		# start solving using EM algorithm
		nIteration = 0

		while(1):

			tic = time.clock(); delta = 0.0

			### M step ###
			# array combined with qZ and Data
			observed = self.qZ*np.swapaxes(np.tile(self.data[:,:].T,[self.nTopics,1,1]),0,2)

			# update pZ
			self.pZ[:] = observed.sum(axis=1).sum(axis=0)
			self.pZ /= self.pZ.sum()

			# update pW
			self.pW[:,:] = observed.sum(axis=0).T
			self.pW /= self.pW.sum(axis=1).reshape((self.nTopics,1))

			# update pD
			self.pD[:,:] = observed.sum(axis=1).T
			self.pD /= self.pD.sum(axis=1).reshape((self.nTopics,1))

			# update eta
			self.eta[:] = optimize.fmin_cg(self.costFunction,self.eta,fprime=self.gradient,args=(self.data,self.qZ,self.zeta,self.label))

			# update zeta
			for d in range(self.nDocuments):
				self.zeta[d] = np.dot(self.qZ[d,:,:].sum(axis=1),self.data[d,:]) + self.sumMulti(self.data[d,:],self.qZ[d,:,:],self.eta,float(self.data[d,:].sum()))

			### E step ###
			# update qZ
			# concerning pSLI
			self.qZ[:,:,:] = np.tile(self.pZ,[self.nDocuments,self.nVocabulary,1])
			self.qZ *= np.tile(self.pW.T,[self.nDocuments,1,1])
			self.qZ *= np.swapaxes(np.tile(self.pD.T,[self.nVocabulary,1,1]),0,1)
			# concerning logistic term
			for d in range(self.nDocuments):
				lq = np.tile(self.eta*self.label[d]/float(self.data[d,:].sum()),[self.nVocabulary,1])
				lq -= np.tile(1.0/self.zeta[d]*np.exp(self.eta/float(self.data[d,:].sum())),[self.nVocabulary,1])*self.sumMultiOmit(self.data[d,:],self.qZ[d,:,:],self.eta,float(self.data[d,:].sum())).reshape((self.nVocabulary,1))
				self.qZ[d,:,:] *= np.exp(lq)
			self.qZ /= self.qZ.sum(axis=2).reshape((self.nDocuments,self.nVocabulary,1))

			### break if converged ###
			self.theta = self.pD.T*self.pZ.reshape((1,self.nTopics))
			self.theta /= self.theta.sum(axis=1).reshape((self.nDocuments,1))
			if nIteration == 0:
				previousTheta = self.theta
			else:
				delta = (np.abs(self.theta-previousTheta).sum(axis=1)/self.data[:,:].sum(axis=1)).max()
				if (delta<self.epsilon and nIteration>10) or nIteration>1000:
					break
				previousTheta = self.theta

			### print information ###
			toc = time.clock()
			self.heatmap(nIteration)
			print "nIteration=%d, time/iteration=%5f, delta=%3e"%(nIteration,toc-tic,delta)
			print self.eta
			nIteration += 1

	def predict(self,data_unseen):
		# data is required to be given in a two dimensional numpy array (nDocuments,nVocabulary)

		# additional parameters for unseen data
		nDocumentsUnseen = data_unseen.shape[0]

		# define q(z) and p(d|z) for unseen data
		pD_unseen = np.zeros((self.nTopics,nDocumentsUnseen),dtype=np.float64)
		qZ_unseen = np.random.rand(nDocumentsUnseen,self.nVocabulary,self.nTopics)
		qZ_unseen /= qZ_unseen.sum(axis=2).reshape((nDocumentsUnseen,self.nVocabulary,1))

		# start predicting topic distribution for unseen data
		nIteration = 0

		while(1):

			tic = time.clock(); delta = 0.0

			### M step ###
			# array combined with qZ and Data
			observed = qZ_unseen*np.swapaxes(np.tile(data_unseen[:,:].T,[self.nTopics,1,1]),0,2)

			# update pD_unseen
			pD_unseen[:,:] = observed.sum(axis=1).T
			pD_unseen /= pD_unseen.sum(axis=1).reshape((self.nTopics,1))

			### E step ###
			# update qZ_unseen
			qZ_unseen[:,:,:] = np.tile(self.pZ,[nDocumentsUnseen,self.nVocabulary,1])
			qZ_unseen *= np.tile(self.pW.T,[nDocumentsUnseen,1,1])
			qZ_unseen *= np.swapaxes(np.tile(pD_unseen.T,[self.nVocabulary,1,1]),0,1)
			qZ_unseen /= qZ_unseen.sum(axis=2).reshape((nDocumentsUnseen,self.nVocabulary,1))

			### break if converged ###
			self.theta_unseen = pD_unseen.T*self.pZ.reshape((1,self.nTopics))
			self.theta_unseen /= self.theta_unseen.sum(axis=1).reshape((nDocumentsUnseen,1))
			if nIteration == 0:
				previousTheta = self.theta_unseen
			else:
				delta = (np.abs(self.theta_unseen-previousTheta).sum(axis=1)/data_unseen[:,:].sum(axis=1)).max()
				if delta<self.epsilon or nIteration>10000:
					break
				previousTheta = self.theta_unseen

			### print information ###
			toc = time.clock()
			print "nIteration=%d, time/iteration=%5f, delta=%3e"%(nIteration,toc-tic,delta)
			nIteration += 1

	def heatmap(self,nIteration):
		# save heatmap image of topic-word distribution
		topicWordDistribution = self.pW/self.pW.sum(axis=1).reshape((self.nTopics,1))

		plt.clf()
		fig,ax = plt.subplots()

		# visualize topic-word distribution
		X,Y = np.meshgrid(np.arange(topicWordDistribution.shape[1]+1),np.arange(topicWordDistribution.shape[0]+1))
		image = ax.pcolormesh(X,Y,topicWordDistribution)
		plt.xlim(0,topicWordDistribution.shape[1])
		plt.xlabel("Vocabulary ID")
		plt.ylabel("Topic ID")
		plt.yticks(np.arange(self.nTopics+1)+0.5,self.eta)

		# show colorbar
		divider = make_axes_locatable(ax)
		ax_cb = divider.new_horizontal(size="2%",pad=0.05)
		fig.add_axes(ax_cb)
		plt.colorbar(image,cax=ax_cb)
		figure = plt.gcf()
		figure.set_size_inches(16,12)
		plt.tight_layout()

		# save image to a file
		plt.savefig("visualization/nIteration_%d.jpg"%nIteration,dpi=100)
		plt.close()

	def save(self,name):
		# save object to a file
		with open(name,"wb") as output:
			cPickle.dump(self.__dict__,output,protocol=cPickle.HIGHEST_PROTOCOL)

	def load(self,name):
		# load object from a file
		with open(name,"rb") as input:
			self.__dict__.update(cPickle.load(input))

