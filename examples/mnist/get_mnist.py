#!/usr/bin/python

import h5py
import urllib2
import cPickle, gzip, os
import numpy as np

datafile = 'mnist.pkl.gz'
if (os.path.exists(datafile)):
    print "Dataset exists, skipping download..."
else:
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'

    file_name = url.split('/')[-1]
    u = urllib2.urlopen(url)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Bytes: %s" % (file_name, file_size)

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print status,
    print('\n')
    f.close()

f = gzip.open('mnist.pkl.gz', 'rb')
train, _, test = cPickle.load(f)
f.close()

inputs = train[0] 
targets = np.zeros((inputs.shape[0], 10), dtype=np.float32)
targets[np.arange(inputs.shape[0]), train[1]] = 0.9
targets += 0.01
filename = h5py.File('train.h5','a')
dset = filename.create_dataset("indata", data=inputs)
dset = filename.create_dataset("outdata", data=targets)
filename.close()
#filehdf5 =  

tmp = np.zeros((test[0].shape[0], 10), dtype=np.float32)
tmp[np.arange(test[0].shape[0]), test[1]] = 0.9
tmp += 0.01
filename = h5py.File('test.h5','a')
dset = filename.create_dataset("indata", data=test[0])
dset = filename.create_dataset("outdata", data=tmp)
filename.close()

