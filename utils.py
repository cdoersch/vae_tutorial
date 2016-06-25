import numpy as np;


def imtile(imlist,width=10,sep=2,brightness=1):
  i=0;
  imrows=[];
  while(i<len(imlist)):
    j=0;
    imrow=[];
    while(j<width):
      imrow.append(imlist[i]);
      j+=1;
      if(j<width):
        imrow.append(np.ones((imlist[i].shape[0],sep))*brightness);
        if(len(imlist[0].shape)==3):
          imrow[-1]=np.tile(imrow[-1][:,:,None],(1,1,imlist[0].shape[2]))
      i+=1;
    imrows.append(np.concatenate(imrow,axis=1))
    if(i<len(imlist)):
      imrows.append(np.ones((sep,imrows[-1].shape[1]))*brightness);
      if(len(imlist[0].shape)==3):
        imrows[-1]=np.tile(imrows[-1][:,:,None],(1,1,imlist[0].shape[2]))
  return np.concatenate(imrows,axis=0);
