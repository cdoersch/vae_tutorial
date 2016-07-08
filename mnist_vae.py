import utils as ut;
import os
import caffe
import caffe.draw
from scipy import misc
import numpy as np
from caffe.proto import caffe_pb2
from google.protobuf import text_format

solver_fn='mnist_vae_solver_adam.prototxt'
net_fn='mnist_vae.prototxt'

try:
  net_proto = caffe_pb2.NetParameter()
  text_format.Merge(open(net_fn).read(), net_proto)
  caffe.draw.draw_net_to_file (net_proto, 'vae_train_net.png', 'TB',
                               phase=caffe.TRAIN)
  caffe.draw.draw_net_to_file (net_proto, 'vae_test_net.png', 'TB',
                               phase=caffe.TEST)
except:
  print("Unable to draw network.  Perhaps your caffe.draw doesn't support phases?");

if not os.path.exists("snapshots"):
  os.makedirs("snapshots")
   
caffe.set_mode_gpu();
solver = caffe.get_solver(solver_fn);

solver.solve()

net=caffe.Net('mnist_vae.prototxt',
              'snapshots/mnist_vae_iter_60000.caffemodel', caffe.TEST)
net.forward()
imlist=[];
for i in range(0,100):
  imlist.append(net.blobs['decode1neuron'].data[i,:].reshape((28,28)));
misc.imsave('vae_out.png',ut.imtile(imlist))
