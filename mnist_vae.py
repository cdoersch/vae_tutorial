import utils as ut;
import os
import caffe
import caffe.draw
from scipy import misc
import numpy as np
#from caffe.proto import caffe_pb2
#from google.protobuf import text_format

solver_fn='mnist_vae_solver_adam.prototxt'
net_fn='mnist_vae.prototxt'

# TODO(cdoersch): make caffe draw_net understand phases
#net_proto = caffe_pb2.NetParameter()
#text_format.Merge(open(net_fn).read(), net_proto)
#caffe.draw.draw_net_to_file (net_proto, 'train_net.png', 'TB')

if not os.path.exists("snapshots"):
  os.makedirs("snapshots")
   
caffe.set_mode_gpu();
solver = caffe.get_solver(solver_fn);

solver.solve()

net=caffe.Net('mnist_vae.prototxt','snapshots/mnist_vae_iter_60000.caffemodel',caffe.TEST)
net.forward()
imlist=[];
for i in range(0,100):
  imlist.append(net.blobs['decode1neuron'].data[i,:].reshape((28,28)));
misc.imsave('vae_out.png',ut.imtile(imlist))
