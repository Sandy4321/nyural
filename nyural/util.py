import h5py
import numpy as np
import os
from ffnet import FFNet
from rnnet import RNNet
from pprint import pprint
from nyural.nonlinearities import (Logistic, Tanh, Softmax, SoftLIF, ReLU,
                                        Continuous, Linear, Nonlinearity,
                                        Gaussian)


def get_dimensions(file_n, dataset):
	ds = "/" + dataset
	f = h5py.File(file_n,'r')
	temp = f[ds]
	dims = temp.shape[1]
	f.close()
	return dims



def prepare_cnf(cnf_,logger):
	cnf = cnf_
	netscheme = cnf_.netscheme
	cnf.numlayers = len(netscheme.train_net.layer) - 1
	# Change for a multi input....
	cnf.source_train = netscheme.train_net.layer[0].hdf5_data_param.source
	cnf.source_test = netscheme.test_net.layer[0].hdf5_data_param.source
	cnf.source_indata = netscheme.train_net.layer[0].top[0]
	cnf.source_target = netscheme.train_net.layer[0].top[1]
    #...
    # Numchunks belongs to solver...
	#cnf.numchunks = int(netscheme.train_net.layer[0].hdf5_data_param.num_chunks)
	#cnf.numchunks_test = int(netscheme.test_net.layer[0].hdf5_data_param.num_chunks)
	#Num cases belongs to net... as batch....
	#cnf.numcases = int(netscheme.train_net.layer[0].hdf5_data_param.batch_size)
	#cnf.numtest = int(netscheme.test_net.layer[0].hdf5_data_param.batch_size)

	#cnf.sizechunk = cnf.numcases/cnf.numchunks
	#cnf.sizechunk_test = cnf.numtest/cnf.numchunks_test

	#with open(cnf.source_train, 'r') as f:
	#	file_n = f.readline().rstrip('\n') 
	layersize = []
	layersize.append(get_dimensions(cnf.source_train,cnf.source_indata))
	layerstype = []
	layerstype.append("Linear")
	error_type = ""

	for i in range (1, cnf.numlayers+1):
		#layers.append(netscheme.train_net.layer[i].type)
		if (netscheme.train_net.layer[i].type == "ReLU"):
			layerstype.append("ReLU")
			layersize.append(netscheme.train_net.layer[i].relu_param.num_output)
		elif (netscheme.train_net.layer[i].type == "Sigmoid"):
			layerstype.append("Logistic")
			layersize.append(netscheme.train_net.layer[i].sigmoid_param.num_output)
		elif (netscheme.train_net.layer[i].type == "Tanh"):
			layerstype.append(Tanh())
			layersize.append(netscheme.train_net.layer[i].tanh_param.num_output)
		elif (netscheme.train_net.layer[i].type == "Softmax"):
			layerstype.append("Softmax")
			layersize.append(netscheme.train_net.layer[i].softmax_param.num_output)
			error_type = netscheme.train_net.layer[i].softmax_param.error_type
		else:
			logger.error("Unknown Type of Layer")
	ff = FFNet(layersize, error_type=error_type, layers=layerstype, use_GPU=bool(cnf.gpu), debug=False) 
	return ff
#ff = FFNet([28 * 28, 1024, 512, 256, 32, 10], error_type="mse",
#   layers=[Linear()] + [ReLU()] * 4 + [Softmax()],
#   use_GPU=True, debug=False)	

def prepare_data(cnf,logger): # TODO: Modify the preparation of the data for each tensor in case of multiple inputs and outputs
#And also the posibility to change the data during the train
	data= empty
	netscheme = cnf.netscheme
	#data.indata = np.zeros((cnf.numcases,cnf.layersize[0]))
	#data.outdata = np.zeros((cnf.numcases,cnf.layersize[cnf.numlayers]))
	#data.intest = np.zeros((cnf.numtest,cnf.layersize[0]))
	#data.outtest = np.zeros((cnf.numtest,cnf.layersize[cnf.numlayers]))
	logger.info("Loading Files")
	
	indata, outdata = get_data(cnf.source_train, cnf.source_indata,cnf.source_target)
	intest, outtest = get_data(cnf.source_test, cnf.source_indata,cnf.source_target)
	test = (intest, outtest)

	#Scaling
	if (netscheme.train_net.layer[0].HasField("transform_param")):
		if (netscheme.train_net.layer[0].transform_param.HasField("mean_file")):
			ds_in = cnf.source_indata + "_minmax"
			ds_out = cnf.source_target + "_minmax"
			filen = netscheme.train_net.layer[0].transform_param.mean_file
			inmean, outmean = get_data(filen,ds_in,ds_out)
			indata_min = inmean[0,:]
			indata_max = inmean[1,:]

			indata = data.indata - indata_min
			indata = data.indata / (indata_max-indata_min)
			indata =  (data.indata * 0.8) + 0.1

			intest = data.intest - indata_min
			intest = data.intest / (indata_max-indata_min)
			intest =  (data.intest * 0.8) + 0.1

			outdata_min = outmean[0,:]
			outdata_max = outmean[1,:]

			outdata = data.outdata - outdata_min
			outdata = data.outdata / (outdata_max-outdata_min)
			outdata =  (data.outdata* 0.8) + 0.1

			outtest = data.outtest - outdata_min
			outtest = data.outtest / (outdata_max-outdata_min)
			outtest =  (data.outtest* 0.8) + 0.1

		elif (netscheme.train_net.layer[0].transform_param.HasField("mean_value")):
			#TODO: Extend Code for mean_value
			mean_value = netscheme.train_net.layer[0].transform_param.mean_value
	
	#Shuffle	

	#train_dx = np.random.permutation(data.indata.shape[0])

	#data.indata = data.indata[train_dx]
	#print type(data.indata)
	#data.outdata = data.outdata[train_dx]
	#print type(data.outdata)

	#test_dx = np.random.permutation(data.intest.shape[0])
	#data.intest = data.intest[test_dx]
	#data.outtest = data.outtest[test_dx]
	return indata, outdata, test


def get_data(filen, ids_, ods_):
	f = h5py.File(filen,'r')

	ids = "/" + ids_
	tmp = f[ids]
	itmp = np.zeros(tmp.shape)
	np.copyto(itmp,tmp)
	itmp = itmp.astype("float32")

	ods = "/" + ods_
	tmp = f[ods]
	otmp = np.zeros(tmp.shape)
	np.copyto(otmp,tmp)
        otmp = otmp.astype("float32")

	f.close()
	return itmp, otmp

def save_model(cnf, ff):
	fname = cnf.snapshot_name + ".h5"
	if (os.path.exists(fname)):
		os.remove(fname)
	modelfile = h5py.File(fname,'a')
	grp = modelfile.create_group("cnf")
	dset = grp.create_dataset("run_name", data=cnf.run_name)
	dset = grp.create_dataset("run_desc", data=cnf.run_desc) 
	#dset = grp.create_dataset("layersize", data=ff.shape)
	#dset = grp.create_dataset("layerstypes", data=', '.join(cnf.layerstypes))

	grp = modelfile.create_group("ff")
	dset = grp.create_dataset("params", data=ff.W)

	modelfile.close()

class empty(object):
    pass
