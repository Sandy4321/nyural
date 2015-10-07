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
import util
import numpy as np
from ffnet import FFNet
from rnnet import RNNet
from nyural.nonlinearities import (Logistic, Tanh, Softmax, SoftLIF, ReLU,
                                        Continuous, Linear, Nonlinearity,
                                        Gaussian)
from optimizers import HessianFree, SGD

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
        self.ff = util.prepare_cnf(self.cnf,self.logger)
        self.inputs, self.targets, self.test = util.prepare_data(self.cnf,self.logger)

    def solve(self):
        cnf = self.cnf
        print cnf.initlambda
        self.ff.run_batches(self.inputs, self.targets, 
                            optimizer= HessianFree(CG_iter=250,init_damping=cnf.initlambda), 
                            batch_size=7500, test=self.test, max_epochs=cnf.max_epoch, 
                            plotting=True)
        self.logger.info("Optimization Done")
        util.save_model(self.cnf, self.ff)

class empty(object):
    pass
