import util
from protoc import nyural_pb2

def NetSolve(netParams):
	train_net = nyural_pb2.NetParameter()
	test_net = nyural_pb2.NetParameter()

	for layer in netParams.layer: 
		
		cnt_trn = 0
		cnt_tst = 0
		if layer.HasField('phase'):
			if (layer.phase == 0):
				train_net.layer.extend([layer])
			if (layer.phase == 1):
				test_net.layer.extend([layer])
		else:
			train_net.layer.extend([layer])
			test_net.layer.extend([layer])

	return train_net, test_net