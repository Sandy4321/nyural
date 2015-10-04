import h5py
import numpy as np
import os
from pprint import pprint

def get_dimensions(file_n, dataset):
	ds = "/" + dataset
	f = h5py.File(file_n,'r')
	temp = f[ds]
	dims = temp.shape[1]
	f.close()
	return dims

def pack_param(W,b,cnf):
	param = np.zeros((cnf.psize,1))
	cur = 0
	for i in range (0,cnf.numlayers):
		param[cur:cur+ cnf.layersize[i]*cnf.layersize[i+1],0] = W[i].flatten()
		cur = cur + cnf.layersize[i]*cnf.layersize[i+1]
		param[cur:cur + cnf.layersize[i+1],0] = b[i].flatten()
		cur = cur + cnf.layersize[i+1]
	return param

def unpack_param(param,cnf):
	weights = []
	biases = []
	cur = 0
	for i in range (0,cnf.numlayers):
		wtmp = np.reshape(param[cur:cur+ cnf.layersize[i]*cnf.layersize[i+1]], (cnf.layersize[i+1],cnf.layersize[i]))
		weights.append(wtmp)
		cur = cur + cnf.layersize[i]*cnf.layersize[i+1]

		btmp = np.reshape(param[cur:cur + cnf.layersize[i+1]], (cnf.layersize[i+1],1))
		biases.append(btmp)
		cur = cur + cnf.layersize[i+1]

	return weights,biases

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
	cnf.numchunks = int(netscheme.train_net.layer[0].hdf5_data_param.num_chunks)
	cnf.numchunks_test = int(netscheme.test_net.layer[0].hdf5_data_param.num_chunks)
	#Num cases belongs to net... as batch....
	cnf.numcases = int(netscheme.train_net.layer[0].hdf5_data_param.batch_size)
	cnf.numtest = int(netscheme.test_net.layer[0].hdf5_data_param.batch_size)

	cnf.sizechunk = cnf.numcases/cnf.numchunks
	cnf.sizechunk_test = cnf.numtest/cnf.numchunks_test

	with open(cnf.source_train, 'r') as f:
		file_n = f.readline().rstrip('\n') 
	layersize = np.zeros((cnf.numlayers+1,1), dtype=np.int)
	layersize[0] = get_dimensions(file_n,cnf.source_indata)

	layers = []
	#Change in range to in layers... netscheme for search layers with updateable parameters and .. in future avoid use of numlayers
	for i in range (1, cnf.numlayers+1):
		layers.append(netscheme.train_net.layer[i].type)
		if (netscheme.train_net.layer[i].HasField("sigmoid_param")):
			layersize[i] = netscheme.train_net.layer[i].sigmoid_param.num_output
		elif (netscheme.train_net.layer[i].HasField("tanh_param")):
			layersize[i] = netscheme.train_net.layer[i].tanh_param.num_output



		#TODO; Continue adding hasfield for all kinds of layers
	psize = np.dot(np.transpose(layersize[0:cnf.numlayers]),layersize[1:cnf.numlayers+1]) + np.sum(layersize[1:cnf.numlayers+1]) 
	
	cnf.layerstypes = layers
	cnf.layersize = layersize
	cnf.psize = int(psize)
	
	maskp = np.ones((cnf.psize,1))
	maskW, maskb = unpack_param(maskp,cnf)
	logger.info("not masking out the weight-decay for biases")
	#for i in range (0,len(maskb)): uncomment this line apply the l_2 only to the connection weights and not the biases
		#maskb{i}(:) = 0: uncomment this line apply the l_2 only to the connection weights and not the biases
	maskp = pack_param(maskW,maskb,cnf)
	cnf.maskp = maskp

	return cnf

def prepare_data(cnf,logger): # TODO: Modify the preparation of the data for each tensor in case of multiple inputs and outputs
#And also the posibility to change the data during the train
	data= empty
	netscheme = cnf.netscheme
	data.indata = np.zeros((cnf.numcases,cnf.layersize[0]))
	data.outdata = np.zeros((cnf.numcases,cnf.layersize[cnf.numlayers]))
	data.intest = np.zeros((cnf.numtest,cnf.layersize[0]))
	data.outtest = np.zeros((cnf.numtest,cnf.layersize[cnf.numlayers]))
	logger.info("Loading Files")

	with open(cnf.source_train, 'r') as f:
		act_ln = 0
		while (act_ln <cnf.numcases):
			file_n = f.readline().rstrip('\n') 
			if not (file_n):
				f.seek(0)
			else:
				itmp, otmp = get_data(file_n, cnf.source_indata,cnf.source_target)
				n_cols = itmp.shape[0]
				act_ln += n_cols
				fs_cl = act_ln - itmp.shape[0]
				if (act_ln > cnf.numcases):
					n_cols = cnf.numcases - fs_cl
				data.indata[fs_cl:act_ln,]=itmp[0:n_cols,]
				data.outdata[fs_cl:act_ln,]=otmp[0:n_cols,]

	with open(cnf.source_test, 'r') as f:
		act_ln = 0
		while (act_ln <cnf.numtest):
			file_n = f.readline().rstrip('\n') 
			if not (file_n):
				f.seek(0)
			else:
				itmp, otmp = get_data(file_n, cnf.source_indata,cnf.source_target)
				n_cols = itmp.shape[0]
				act_ln += n_cols
				fs_cl = act_ln - itmp.shape[0]
				if (act_ln > cnf.numtest):
					n_cols = cnf.numtest - fs_cl
				data.intest[fs_cl:act_ln,]=itmp[0:n_cols,]
				data.outtest[fs_cl:act_ln,]=otmp[0:n_cols,]

	#Scaling
	if (netscheme.train_net.layer[0].HasField("transform_param")):
		if (netscheme.train_net.layer[0].transform_param.HasField("mean_file")):
			ds_in = cnf.source_indata + "_minmax"
			ds_out = cnf.source_target + "_minmax"
			filen = netscheme.train_net.layer[0].transform_param.mean_file
			inmean, outmean = get_data(filen,ds_in,ds_out)
			indata_min = inmean[0,:]
			indata_max = inmean[1,:]

			data.indata = data.indata - indata_min
			data.indata = data.indata / (indata_max-indata_min)
			data.indata =  (data.indata * 0.8) + 0.1

			data.intest = data.intest - indata_min
			data.intest = data.intest / (indata_max-indata_min)
			data.intest =  (data.intest * 0.8) + 0.1

			outdata_min = outmean[0,:]
			outdata_max = outmean[1,:]

			data.outdata = data.outdata - outdata_min
			data.outdata = data.outdata / (outdata_max-outdata_min)
			data.outdata =  (data.outdata* 0.8) + 0.1

			data.outtest = data.outtest - outdata_min
			data.outtest = data.outtest / (outdata_max-outdata_min)
			data.outtest =  (data.outtest* 0.8) + 0.1

		elif (netscheme.train_net.layer[0].transform_param.HasField("mean_value")):
			mean_value = netscheme.train_net.layer[0].transform_param.mean_value
	
	#Shuffle	

	train_dx = np.random.permutation(data.indata.shape[0])

	data.indata = data.indata[train_dx]
	data.outdata = data.outdata[train_dx]

	test_dx = np.random.permutation(data.intest.shape[0])
	data.intest = data.intest[test_dx]
	data.outtest = data.outtest[test_dx]
	return data

def prepare_info(cnf,logger): #TODO: improve the preparation in case of finetuning.. loading parameters from a previous trained network
	logger.info("Setting up Parameters")
	info = empty
	ch = np.zeros((cnf.psize,1))
	lmdba = cnf.initlambda

	llrecord = np.zeros((cnf.max_epoch,2))
	errrecord = np.zeros((cnf.max_epoch,2))
	lambdarecord = np.zeros((cnf.max_epoch,1))
	times = np.zeros((cnf.max_epoch,1))

	totalpasses = 0
	epoch = 0

	paramsp = np.zeros((cnf.psize,1))
	Wtmp,Btmp = unpack_param(paramsp,cnf)

	numconn = 15
	for i in range (0,cnf.numlayers):
		initcoeff = 1
		
		if ( (i > 0) and (cnf.layerstypes[i-1] == "tanh")):
			initcoeff = 0.5*initcoeff
		if (cnf.layerstypes[i] == "tanh"):
			initcoeff = 0.5*initcoeff

			Btmp[i][:,0] = 0.5#*np.ones(Btmp[i].shape)
		for j in range (0,cnf.layersize[i+1]):
			idx = np.ceil((cnf.layersize[i]-1)*np.random.rand(numconn)).astype(int)
			Wtmp[i][j,idx] = np.random.rand(numconn)*initcoeff
	paramsp = pack_param(Wtmp,Btmp,cnf)
	info.paramsp = paramsp
	info.ch = ch
	info.epoch = epoch
	info.lmdba = lmdba
	info.totalpasses = totalpasses
	info.llrecord = llrecord
	info.times = times
	info.errrecord = errrecord
	info.lambdarecord = lambdarecord
	return info

def get_data(filen, ids_, ods_):
	f = h5py.File(filen,'r')

	ids = "/" + ids_
	tmp = f[ids]
	itmp = np.zeros(tmp.shape)
	np.copyto(itmp,tmp)

	ods = "/" + ods_
	tmp = f[ods]
	otmp = np.zeros(tmp.shape)
	np.copyto(otmp,tmp)

	f.close()
	return itmp, otmp

def save_model(cnf, info):
	fname = cnf.snapshot_name + ".h5"
	if (os.path.exists(fname)):
		os.remove(fname)
	modelfile = h5py.File(fname,'a')
	grp = modelfile.create_group("cnf")
	dset = grp.create_dataset("run_name", data=cnf.run_name)
	dset = grp.create_dataset("run_desc", data=cnf.run_desc) 
	dset = grp.create_dataset("layersize", data=cnf.layersize)
	dset = grp.create_dataset("layerstypes", data=', '.join(cnf.layerstypes))
	dset = grp.create_dataset("psize", data=cnf.psize)

	grp = modelfile.create_group("info")
	dset = grp.create_dataset("params", data=info.paramsp)
	dset = grp.create_dataset("lmdba", data=info.lmdba)
	dset = grp.create_dataset("ch", data=info.ch)
	dset = grp.create_dataset("llrecord", data=info.llrecord)
	dset = grp.create_dataset("errrecord", data=info.errrecord)
	dset = grp.create_dataset("times", data=info.times)
	modelfile.close()
class empty(object):
    pass
