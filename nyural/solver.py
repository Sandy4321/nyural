#
# * Copyright (C) 2015 Kuniaki Noda (kuniaki.noda@gmail.com)
# * This is UNPUBLISHED PROPRIETARY of Kuniaki Noda;
# * the contents of this file is not to be disclosed to third parties, copied
# * or duplicated in any form, in whole or in part, without the prior written
# * permission of Kuniaki Noda.
#
#import utilweightcost

import net
import time
import optim
import util
import numpy as np
from protoc import nyural_pb2

from pprint import pprint



class solver(object):
	def __init__(self, solverParam, logger):
		cnf= empty()
		self.logger = logger
		message = "Setting Solver Parameters" + "\n"
		
		if solverParam.HasField('seed'):
			np.random.seed(solverParam.seed)
		else:
			np.random.seed(1234)
		
		cnf.run_desc = solverParam.run_desc
		message += "run_desc: " + str(cnf.run_desc) + "\n"
		cnf.run_name = solverParam.run_name
		message += "run_name: " + str(cnf.run_name) + "\n"
		cnf.max_epoch = solverParam.max_iter
		message += "max epoch: " + str(cnf.max_epoch) + "\n"
		cnf.decay = solverParam.decay 
		message += "decay: " + str(cnf.decay) + "\n"
		cnf.weightcost = solverParam.weightcost
		message += "weightcost: " + str(cnf.weightcost)+ "\n"
		cnf.autodamp = solverParam.autodamp
		message += "autodamp: " + str(cnf.autodamp) + "\n"
		cnf.drop = solverParam.drop
		message += "drop: " + str(cnf.drop) + "\n"
		cnf.boost = solverParam.boost
		message += "boost: " + str(cnf.boost)+ "\n"
		cnf.rms = solverParam.rms
		message += "rms: " + str(cnf.rms)+ "\n"
		cnf.errtype = solverParam.errtype
		message += "errtype: " + str(cnf.errtype)+ "\n"
		cnf.initlambda = solverParam.initlambda 
		message += "initlambda: " + str(cnf.initlambda) + "\n"
		cnf.mattype = solverParam.mattype
		message += "mattype: " + str(cnf.mattype)+ "\n"
		cnf.skip = solverParam.skip
		cnf.hibrid = solverParam.hibrid
		message += "hibrid: " + str(cnf.hibrid)+ "\n"
		cnf.snapshot_name = solverParam.snapshot_name
		message += "snapshot_name: " + str(cnf.snapshot_name) + "\n"
		cnf.gpu = solverParam.gpu
		message += "gpu: " + str(cnf.gpu) 
		self.cnf = cnf
		self.logger.info(message)
		
	#TODO: Need to arrange data	

	def InitNet(self, netParam):
		netscheme = empty()
		netscheme.train_net, netscheme.test_net = net.NetSolve(netParam)
		message = "Training Network:" + "\n"
		message += str(netscheme.train_net)
		self.logger.info(message)

		message = "Testing Network:" + "\n"
		message += str(netscheme.test_net)
		self.logger.info(message)
		self.cnf.netscheme = netscheme

	def InitParams(self):
		cnf = util.prepare_cnf(self.cnf,self.logger)
		info = util.prepare_info(cnf,self.logger)
		data = util.prepare_data(cnf,self.logger)
		self.cnf = cnf
		self.info = info
		self.data = data

	def solve(self):
		cnf = self.cnf
		for i in range (0,cnf.max_epoch):
			self.epoch(i)		
			if (i > 50) and (self.info.errrecord[-10,1] < self.info.errrecord[-1,1]):
				self.logger("Overfitting detected, terminating")
				break
		self.logger.info("Saving model in: " + cnf.snapshot_name) #TODO: move this into the for and save according to a specific period
		util.save_model(cnf,self.info)


	def epoch(self,epoch):
		data = self.data
		info = self.info

		self.logger.info("epoch: " + str(epoch) )
		start = time.time()
		targetchunk = np.remainder(epoch, self.cnf.numchunks)

		#-------- calc gradient
		hf_optim = optim.hf_optimizer(self.cnf, data, info, targetchunk, self.logger)
		hf_optim.calc_gradient()

		#-------- update parameter
		hf_optim.preconditioning()
		info, ll, err = hf_optim.optimize();

		#-------- evaluate result
		ll_test, err_test = hf_optim.evaluate(data.intest, data.outtest, self.cnf.numchunks_test);
		
		#-------- update info
		info.epoch = epoch
		info.lambdarecord[epoch,0] = info.lmbda;
		info.llrecord[epoch,0] = ll;
		info.llrecord[epoch,1] = ll_test;
		info.errrecord[epoch,0] = err;
		info.errrecord[epoch,1] = err_test;
		info.times[epoch] = time.time()-start

		self.info = info

		#-------- print status
		message = "epoch results \n"
		message += "TRAIN Log likelihood: {: 2.5f}, error rate: {: 2.5f}\n".format(ll, err)
		message += "TEST  Log likelihood: {: 2.5f}, error rate:  {: 2.5f}\n".format(ll_test,err_test)
		message += "Error rate difference (test - train): {: 2.5f}\n".format(err_test-err);
		message += "Computation time: {:3.4f}[sec] \n".format(info.times[epoch,0])
		message += "Total time: {:3.4f}[min]".format(np.sum(info.times)/60)
		self.logger.info(message);

class empty(object):
    pass
