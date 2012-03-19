import cPickle, math, os, os.path, sys, time

import scipy

from theano.tensor.shared_randomstreams import RandomStreams as RandomStreamsCPU
from theano.sandbox.rng_mrg  import MRG_RandomStreams as RandomStreamsGPU
import theano
import theano.tensor as T
import theano.sparse
#import extra_theano_ops as theano_sparse

import numpy, math, cPickle, sys, gzip
from numpy.random import shuffle



#data_path = '/mnt/scratch/bengio/glorotxa/data/'
#c_path = '/home/glorotxa/'

trainset_path = '/data/lisa/data/UTLC/sparse/terry_train.npy.gz'
validset_path = '/data/lisa/data/UTLC/sparse/terry_valid.npy.gz'
testset_path = '/data/lisa/data/UTLC/sparse/terry_test.npy.gz'

def binomial_NLP_noise(inp,zer_mask,one_mask):
    return zer_mask * inp + (inp<1) * one_mask
    #return theano_rng.binomial( size = inp.shape, n = 1, p =  1 - noise_lvl[0], dtype=theano.config.floatX) * inp \
    #                    + (inp==0) * theano_rng.binomial( size = inp.shape, n = 1, p =  noise_lvl[1], dtype=theano.config.floatX)

def cross_entropy_sampled_cost(target, output_act,pattern,scaling = False):
    XE = target * (- T.log(1 + T.exp(-output_act))) + (1 - target) * (- T.log(1 + T.exp(output_act)))
    if pattern != None:    
        XE = XE * pattern
    if scaling != False:
        XE = XE * T.cast(scaling[0],dtype=theano.config.floatX) * target + XE * T.cast(scaling[1],dtype=theano.config.floatX) * (1-target)
    return -T.mean(T.sum(XE, axis=1),axis=0), -T.mean(XE,axis=0)

def MSE_sampled_cost(target, output_act,pattern,scaling = False):
    MSE = (T.nnet.sigmoid(output_act) - target) * (T.nnet.sigmoid(output_act) - target)
    if pattern != None:
        MSE = MSE * pattern
    if scaling != False:
        MSE = MSE * T.cast(scaling[0],dtype=theano.config.floatX) * target + MSE * T.cast(scaling[1],dtype=theano.config.floatX) * (1-target)
    return T.mean(T.sum(MSE, axis=1),axis=0), T.mean(MSE,axis=0)


def vectosparsemat(path,NBDIMS):
    """
    This function converts the unlabeled training data into a scipy
    sparse matrix and return it.
    """
    print >> sys.stderr , "Read and converting data file: %s to a sparse matrix"%path 
    # We first count the number of line in the file
    f = open(path, 'r')
    i = f.readline()
    ct = 0
    while i!='':
        ct+=1
        i = f.readline()
    f.close()
    # Then we allocate and fill the sparse matrix as a lil_matrix 
    # for efficiency.
    NBEX = ct
    train = scipy.sparse.lil_matrix((NBEX,NBDIMS),dtype=theano.config.floatX)
    f = open(path, 'r')
    i = f.readline()
    ct = 0
    next_print_percent = 0.1
    while i !='':
        if ct / float(NBEX) > next_print_percent:
            print >> sys.stderr , "\tRead %s %s of file"%(next_print_percent*100,'%')
            next_print_percent += 0.1
        i = i[:-1]
        i = list(i.split(' '))
        for j in i:
            if j!='':
                idx,dum,val = j.partition(':')
                train[ct,int(idx)-1] = 1
        i = f.readline()
        ct += 1
    print >> sys.stderr , "Data converted" 
    # We return a csr matrix because for efficiency 
    # because we will later shuffle the rows.
    return train.tocsr()


def createdensebatch(spmat,size,batchnumber):
    """
    not in use
    
    This function creates and return dense matrix corresponding to the 
    'batchnumber'_th slice of length 'size' of the sparse data matrix.
    """
    NB_DENSE = int(numpy.ceil(spmat.shape[0] / float(size)))
    assert batchnumber>=0 and batchnumber<NB_DENSE 
    realsize = size
    if batchnumber< NB_DENSE-1:
        batch = numpy.asarray(spmat[size*batchnumber:size*(batchnumber+1),:].toarray(),dtype=theano.config.floatX)
    else:
        batch = numpy.asarray(spmat[size*batchnumber:,:].toarray(),dtype=theano.config.floatX)
        realsize = batch.shape[0]
        if batch.shape[0] < size:
            batch = numpy.concatenate([batch,numpy.zeros((size-batch.shape[0],batch.shape[1]),dtype=theano.config.floatX)])
    return batch,realsize
 

def createvecfile(Wenc,benc,PathData,depth,OutFile,act = 'rect'):
    """
    This function builds a 'OutFile' .vec file corresponding to 'PathData' taken at 
    the layer 'depth' of the 'PathLoad' model.
    """
    print >> sys.stderr, "Creating vec file %s ( depth=%d, datafiles=%s)..." % (repr(OutFile), depth,PathData)
    inp = T.matrix()
    hid_lin = T.dot(inp,Wenc)+benc
    if act == 'rect':
        hid_out = hid_lin * (hid_lin>=0)
    if act == 'sigmoid':
        hid_out = T.nnet.sigmoid(hid_lin)
    outputs = [hid_out]
    func = theano.function([inp],outputs)

    full_train = vectosparsemat(PathData,Wenc.value.shape[0])
    NB_BATCHS = int(numpy.ceil(full_train.shape[0] / float(500)))

    f = open(OutFile,'w')

    for i in range(NB_BATCHS):
        if i < NB_BATCHS-1:
            rep = func(numpy.asarray(full_train[500*i:500*(i+1),:].toarray(),dtype=theano.config.floatX))[0]
        else:
            rep = func(numpy.asarray(full_train[500*i:,:].toarray(),dtype=theano.config.floatX))[0]
        textr = ''
        for l in range(rep.shape[0]):
            idx = rep[l,:].nonzero()[0]
            for j,v in zip(idx,rep[l,idx]):
                textr += '%s:%s '%(j,v)
            textr += '\n'
        f.write(textr)
    f.close()
    print >> sys.stderr, "...done creating vec files"


def createvecfilesparse(Wenc,benc,data,OutFile1,OutFile2,act = 'rect'):
    """
    This function builds a 'OutFile' .vec file corresponding to 'PathData' taken at 
    the layer 'depth' of the 'PathLoad' model.
    """
    inp = theano.sparse.csr_matrix()
    hid_lin = theano.sparse.dot(inp,Wenc)+benc
    if act == 'rect':
        hid_out = hid_lin * (hid_lin>=0)
    if act == 'sigmoid':
        hid_out = T.nnet.sigmoid(hid_lin)
    outputs = [hid_out]
    func = theano.function([inp],outputs)
    rep1 = func(data)[0]
    rep2 = numpy.dot(rep1,rep1.T)
    if act == 'sigmoid':
        rep1 = numpy.floor((rep1 / rep1.max())*999)
        rep2 = numpy.floor((rep2 / rep2.max())*999)
    else:
        rep1 = numpy.floor(uniformisation(rep1)*999)
        rep2 = numpy.floor(uniformisation(rep2)*999)
    f = open(OutFile1,'w')
    g = open(OutFile2,'w')
    txt1=''
    txt2=''
    for i in range(rep1.shape[0]):
        for j in range(rep1.shape[0]):
            txt2 += '%s '%int(rep2[i,j])
        for j in range(rep1.shape[1]):
            txt1 += '%s '%int(rep1[i,j])
        txt1 += '\n'
        txt2 += '\n'
    f.write(txt1)
    g.write(txt2)
    g.close()
    f.close()
    print >> sys.stderr, "...done creating vec files"


def createWbshared(rng,n_inp,n_hid,tag,trans=False):
    #wbound = numpy.sqrt(6./(n_inp+n_hid))
    wbound = numpy.sqrt(6./(5000.))
    if not(trans):
        W_values = numpy.asarray( numpy.random.uniform( low = -wbound, high = wbound, \
                                    size = (n_inp, n_hid)), dtype = theano.config.floatX)
    else:
        W_values = numpy.asarray( numpy.random.uniform( low = -wbound, high = wbound, \
                                    size = (n_hid, n_inp)), dtype = theano.config.floatX)
    W = theano.shared(value = W_values, name = 'W'+tag)
    b_values = numpy.zeros((n_hid,), dtype= theano.config.floatX)
    b = theano.shared(value= b_values, name = 'b'+tag)
    return W,b


def uniformisation(inparray,zer = True):
    "Exact uniformization of the inparray (matrix) data"
    # Create ordered list of elements
    listelem = list(numpy.sort(list(set(inparray.flatten()))))
    dictP = {}
    totct = 0
    outarray = numpy.ones_like(inparray)
    #initialize element count
    for i in listelem:
        dictP.update({i:0})
    #count
    for i in range(inparray.shape[0]):
        if len(inparray.shape) == 2:
            for j in range(inparray.shape[1]):
                dictP[inparray[i,j]]+=1
                totct +=1
        else:
            dictP[inparray[i]]+=1
            totct +=1
    #cumulative
    prev = 0
    for i in listelem:
        dictP[i]+= prev
        prev = dictP[i]
    #conversion
    for i in range(inparray.shape[0]):
        if len(inparray.shape) == 2:
            for j in range(inparray.shape[1]):
                outarray[i,j] = dictP[inparray[i,j]]/float(totct)
        else:
            outarray[i] = dictP[inparray[i]]/float(totct)
    if zer:
        outarray = outarray - dictP[listelem[0]]/float(totct)
        outarray /= outarray.max()
    return outarray



def SamplingsparseSDAEexp(state,channel):
    """
    This script launch a SDAE experiment, training in a greedy layer wise fashion.
    The hidden layer activation function is the rectifier activation (i.e. max(0,y)). The reconstruction activation function
    is the sigmoid. The reconstruction cost is the cross-entropy. From one layer to the next we need to scale the
    parameters in order to ensure that the representation is in the interval [0,1].
    The noise of the input layer is a salt and pepper noise ('binomial_NLP'), for deeper layers it is a zero masking
    noise (binomial).
    """
    SavePath = channel.remote_path+'/' if hasattr(channel,'remote_path') else channel.path+'/'
    numpy.random.seed(state.seed)
    
    if hasattr(state,'featsub'):
        # sub
        listfeat = numpy.load(open(state.featsub))
        state.ninputs = len(listfeat)
        state.n_inp = len(listfeat)
    
    Wenc,benc = createWbshared(numpy.random,state.n_inp,state.n_hid,'enc')
    Wdec,bdec = createWbshared(numpy.random,state.n_hid,state.n_inp,'dec',trans = True)
    
    # Load the entire training data
    full_train = scipy.sparse.csr_matrix(numpy.load(gzip.open(trainset_path)), dtype=theano.config.floatX)
    if state.trans == 'test':
        full_valid = scipy.sparse.csr_matrix(numpy.load(gzip.open(validset_path)), dtype=theano.config.floatX)[1:]
        full_test = scipy.sparse.csr_matrix(numpy.load(gzip.open(testset_path)), dtype=theano.config.floatX)[1:]
    else: 
        full_test = scipy.sparse.csr_matrix(numpy.load(gzip.open(validset_path)), dtype=theano.config.floatX)[1:]
        full_valid = scipy.sparse.csr_matrix(numpy.load(gzip.open(testset_path)), dtype=theano.config.floatX)[1:]
    
    if not hasattr(state,'con'):
        full_train.data = numpy.ones_like(full_train.data)
        full_valid.data = numpy.ones_like(full_valid.data)
        full_test.data = numpy.ones_like(full_test.data)
    else:
        if state.con == True:
            all = uniformisation(numpy.concatenate([full_train.data,full_valid.data,full_test.data]),False)
            full_train.data = all[:-(len(full_valid.data)+len(full_test.data))]
            full_valid.data = all[-(len(full_valid.data)+len(full_test.data)):-(len(full_test.data))]
            full_test.data = all[-(len(full_test.data)):]
    
    full_train = scipy.sparse.vstack([full_train,full_valid],'csr')
    for i in range(state.N):
        full_train = scipy.sparse.vstack([full_train,full_test],'csr')
    full_train = full_train[numpy.random.permutation(full_train.shape[0]),:]
    if hasattr(state,'featsub'):
        full_train = full_train[:,listfeat]
        full_test = full_test[:,listfeat]
        full_valid = full_valid[:,listfeat]
    if hasattr(state,'con') and state.con=='white':
        stdall = numpy.std(full_train.data)
        full_train.data = full_train.data / stdall
        full_test.data = full_test.data / stdall
        full_valid.data = full_valid.data / stdall
    #------------------------------
    state.bestonlinerec = -1
    state.bestonlinerecde = -1
    epochsl = []
    reconline = []
    #-------------------------------

    # Model initialization:

    inp = theano.sparse.csr_matrix()
    pattern = T.matrix()
    target = T.matrix()
    
    hid_lin = theano.sparse.dot(inp,Wenc)+benc
    if state.act == 'rect':
        hid_out = hid_lin * (hid_lin > 0)
    if state.act == 'sigmoid':
        hid_out = T.nnet.sigmoid(hid_lin)
    L1_reg = T.mean(T.sum(hid_out * hid_out,axis=1),axis=0)
    rec_lin = theano.sparse.sampling_dot( hid_out, Wdec , pattern)+bdec
    #rec_lin = extra_theano_ops.sampling_dot( hid_out, Wdec , pattern)+bdec
    # the sigmoid is inside the cross_entropy function.
    if not hasattr(state,'scaling'):
        state.scaling = False
    if state.cost == 'CE':
        cost, dum = cross_entropy_sampled_cost(target, rec_lin,pattern,state.scaling)
        cost_dense, cost_decoupled_dense = cross_entropy_sampled_cost(target, T.dot(hid_out,Wdec.T) +bdec,None)
    if state.cost == 'MSE':
        cost, dum = MSE_sampled_cost(target, rec_lin,pattern,state.scaling)
        cost_dense, cost_decoupled_dense = MSE_sampled_cost(target, T.dot(hid_out,Wdec.T)+bdec ,None)
    if state.regcoef != 0.:
        cost = cost + state.regcoef * L1_reg

    grad = T.grad(cost,[Wenc,Wdec,benc,bdec])
    updates = dict( (p,p-state.lr*g) for p,g in zip([Wenc,Wdec,benc,bdec],grad) )
    TRAINFUNC = theano.function([inp,target,pattern],cost, updates = updates)
    ERR = theano.function([inp,target],[cost_dense,cost_decoupled_dense]) 
    
    # Train the current DAE
    for epoch in range(state['nepochs']):
    # Load sequentially dense batches of the training data
        reconstruction_error_batch = 0
        update_count1 = 0
        for batchnb in range(full_train.shape[0]/state.batchsize):
            tmpinp = numpy.asarray(full_train[batchnb*state.batchsize:(batchnb+1)*state.batchsize].toarray(),dtype=theano.config.floatX)
            zer = numpy.asarray(numpy.random.binomial(n = 1, p = 1-state.zeros, size = tmpinp.shape),dtype=theano.config.floatX)
            tmpinpnoise = scipy.sparse.csr_matrix(zer * tmpinp,dtype=theano.config.floatX)
            if state.pattern == 'inp':
                tmppattern = numpy.asarray((tmpinp + numpy.random.binomial(size = tmpinp.shape,n=1,p=state.ratio))>0,dtype=theano.config.floatX )
            elif state.pattern == 'noise':
                tmppattern = numpy.asarray(((1-zer)*tmpinp + numpy.random.binomial(size = tmpinp.shape,n=1,p=state.ratio))>0,dtype=theano.config.floatX)
            elif state.pattern == 'inpnoise':
                tmppattern = numpy.asarray((tmpinp + numpy.random.binomial(size = tmpinp.shape,n=1,p=state.ratio))>0,dtype=theano.config.floatX)
            elif state.pattern == 'random':
                tmppattern = numpy.asarray(numpy.random.binomial(size = tmpinp.shape,n=1,p=state.ratio),dtype=theano.config.floatX)
            tmp = TRAINFUNC(tmpinpnoise,tmpinp,tmppattern) 
	    reconstruction_error_batch += tmp
            update_count1 += 1
        print >> sys.stderr, '...finished training epoch #%s' % (epoch+1)
	print >> sys.stderr, "\t\tMean reconstruction error %s" % (reconstruction_error_batch/float(update_count1))
        full_train = full_train[numpy.random.permutation(full_train.shape[0]),:]
        if epoch+1 in state.epochs:
            #rec test err
	    if not os.path.isdir(SavePath):
	        os.mkdir(SavePath)
            if hasattr(state,'savespec'):
                modeldir = os.path.join(SavePath, 'epoch%s'%(epoch+1) )
            else:
                modeldir = os.path.join(SavePath, 'currentmodel' )
	    if not os.path.isdir(modeldir):
	        os.mkdir(modeldir)
            f = open(modeldir+'/params.pkl','w')
            cPickle.dump(Wenc.value,f,-1)
            cPickle.dump(Wdec.value,f,-1)
            cPickle.dump(benc.value,f,-1)
            cPickle.dump(bdec.value,f,-1)
            f.close()
            if state.trans == 'test':
	        createvecfilesparse(Wenc,benc,full_valid,SavePath+'/terry_dl%s_valid.prepro'%(epoch+1),SavePath+'/terry_sdl%s_valid.prepro'%(epoch+1),state.act)
                createvecfilesparse(Wenc,benc,full_test,SavePath+'/terry_dl%s_final.prepro'%(epoch+1),SavePath+'/terry_sdl%s_final.prepro'%(epoch+1),state.act)
                os.system('zip %s %s %s'%(SavePath+'/terry_dl%s.zip'%(epoch+1),SavePath+'/terry_dl%s_valid.prepro'%(epoch+1),SavePath+'/terry_dl%s_final.prepro'%(epoch+1)))
                os.system('zip %s %s %s'%(SavePath+'/terry_sdl%s.zip'%(epoch+1),SavePath+'/terry_sdl%s_valid.prepro'%(epoch+1),SavePath+'/terry_sdl%s_final.prepro'%(epoch+1)))
            else:
                createvecfilesparse(Wenc,benc,full_test,SavePath+'/terry_dl%s_valid.prepro'%(epoch+1),SavePath+'/terry_sdl%s_valid.prepro'%(epoch+1),state.act)
                createvecfilesparse(Wenc,benc,full_valid,SavePath+'/terry_dl%s_final.prepro'%(epoch+1),SavePath+'/terry_sdl%s_final.prepro'%(epoch+1),state.act)
                os.system('zip %s %s %s'%(SavePath+'/terry_dl%s.zip'%(epoch+1),SavePath+'/terry_dl%s_valid.prepro'%(epoch+1),SavePath+'/terry_dl%s_final.prepro'%(epoch+1)))
                os.system('zip %s %s %s'%(SavePath+'/terry_sdl%s.zip'%(epoch+1),SavePath+'/terry_sdl%s_valid.prepro'%(epoch+1),SavePath+'/terry_sdl%s_final.prepro'%(epoch+1)))
            epochsl += [epoch+1] 
            reconline += [reconstruction_error_batch/float(update_count1)]
	    print '###### RESULTS :'
	    print 'Depth:',1
	    print 'Epoch:',epoch+1
	    print 'Online Reconstruction:',reconstruction_error_batch/float(update_count1)
	    print ' '
	    f = open('results.pkl','w')
	    cPickle.dump(epochsl,f,-1)
            cPickle.dump(reconline,f,-1)
            f.close()
	    if reconstruction_error_batch/float(update_count1) < state.bestonlinerec  or state.bestonlinerec==-1:
	        state.bestonlinerec = reconstruction_error_batch/float(update_count1)
	        state.bestonlinerecde =(1,epoch+1)
	state.currentepoch = epoch+1
    return channel.COMPLETE
