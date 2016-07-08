import utils as ut;
import os
import caffe
import caffe.draw
from scipy import misc
import numpy as np
from caffe.proto import caffe_pb2
from google.protobuf import text_format

solver_fn='mnist_cvae_solver_adam.prototxt'
net_fn='mnist_cvae.prototxt'

try:
  net_proto = caffe_pb2.NetParameter()
  text_format.Merge(open(net_fn).read(), net_proto)
  caffe.draw.draw_net_to_file (net_proto, 'cvae_train_net.png', 'TB',
                               phase=caffe.TRAIN)
  caffe.draw.draw_net_to_file (net_proto, 'cvae_test_net.png', 'TB',
                               phase=caffe.TEST)
except:
  print("Unable to draw network.  Perhaps your caffe.draw doesn't support phases?");

if not os.path.exists("snapshots"):
  os.makedirs("snapshots")
   
caffe.set_mode_gpu();
solver = caffe.get_solver(solver_fn);

solver.solve()

# The rest of the file is displaying the output

net=caffe.Net('mnist_cvae.prototxt','snapshots/mnist_cvae_iter_60000.caffemodel',caffe.TEST)
net.forward()

# The network uses sampling to binarize the input values.  However, there's no
# simple way to make that sampling deterministic, which means it's difficult to
# compare the results to the regressor.  Hence, we re-initialize the inputs in
# python so that they can be deterministic.
np.random.seed(1)
unifa=np.random.uniform(0,2,size=net.blobs['uniform'].data.shape);
net.blobs['uniform'].data[:]=unifa
net.forward(start='dataconcat')
imlist=[];
origlist=[]
for i in range(0,15*4): 
  imgen=net.blobs['decode1neuron'].data[i,:].reshape((28,27))
  imgen=np.tile(imgen[:,:,None],(1,1,3))
  iminp=net.blobs['data_right'].data[i,0,:,:]
  iminp=iminp[:,:,None]
  iminp=np.concatenate([iminp,iminp*.5,1-iminp],axis=2)
  imlist.append(np.concatenate((imgen[:,0:13,:],iminp*.5+.5,imgen[:,14:,:]),axis=1));
  origlist.append(net.blobs['data'].data[i,0,:,:])
  # scale up the output to make sure the pixels are clearly visible.
  imlist[-1]=misc.imresize(imlist[-1],[28*10,28*10],'nearest')
  origlist[-1]=misc.imresize(origlist[-1],[28*10,28*10],'nearest')
print('saving results...')
misc.imsave('cvae.png',ut.imtile(imlist,width=15,sep=20,brightness=255))
misc.imsave('groundtruth.png',ut.imtile(origlist,width=15,sep=20,brightness=255))
