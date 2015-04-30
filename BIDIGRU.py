import numpy
import theano
import sys
import math
from theano import tensor as T
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict


def build_shared_zeros(shape, name):
    """ Builds a theano shared variable filled with a zeros numpy array """
    return shared(value=numpy.zeros(shape, dtype=theano.config.floatX),
                name=name, borrow=True)


def np_ortho(shape, rng, name=None):
    """ Builds a theano variable filled with orthonormal random values """
    g = rng.randn(*shape)
    o_g = linalg.svd(g)[0]
    return o_g.astype(theano.config.floatX)


def shared_ortho(shape, rng, name=None):
    """ Builds a theano shared variable filled with random values """
    g = rng.randn(*shape)
    o_g = linalg.svd(g)[0]
    return theano.shared(value=o_g.astype(theano.config.floatX), borrow=True)


def orthogonal(shape, scale=1.1):
    """ benanne lasagne ortho init (faster than qr approach)"""
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v # pick the one with the correct shape
    q = q.reshape(shape)
    return shared(scale * q[:shape[0], :shape[1]] , )


def softplus_f(v):
    """activation for a softplus layer, not here"""
    return T.log(1 + T.exp(v))

def dropout(rng, x, p=0.5):
    """ Zero-out random values in x with probability p using rng """
    if p > 0. and p < 1.:
        seed = rng.randint(2 ** 30)
        srng = theano.tensor.shared_randomstreams.RandomStreams(seed)
        mask = srng.binomial(n=1, p=1.-p, size=x.shape,
                dtype=theano.config.floatX)
        return x * mask
    return x
 
def fast_dropout(rng, x):
    """ Multiply activations by N(1,1) """
    seed = rng.randint(2 ** 30)
    srng = RandomStreams(seed)
    mask = srng.normal(size=x.shape, avg=1., dtype=theano.config.floatX)
    return x * mask

def dropoutv(X, p=0.):
    if p != 0:
        retain_prob = 1 - p
        X = X / retain_prob * srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
    return X


def relu_f(vec):
    """ Wrapper to quickly change the rectified linear unit function """
    return (vec + abs(vec)) / 2.

def clipped_relu(vec): #Clipped ReLU from DeepSpeech paper
    """ min{max{0, z}, 20} is the clipped rectified-linear (ReLu) activation function """
    #original return (vec + abs(vec)) / 2.
    #return np.minimum(np.maximum(0,vec),20)
    return T.clip(vec,0,20)
    #100 loops, best of 3: 3.44 ms per loop

def leaky_relu_f(z):
    return T.switch(T.gt(z, 0), z, z * 0.01)
    #100 loops, best of 3: 7.11 ms per loop


def _make_ctc_labels(y):
    # Assume that class values are sequential! and start from 0
    highest_class = np.max([np.max(d) for d in y])
    # Need to insert blanks at start, end, and between each label
    # See A. Graves "Supervised Sequence Labelling with Recurrent Neural
    # Networks" figure 7.2 (pg. 58)
    # (http://www.cs.toronto.edu/~graves/preprint.pdf)
    blank = highest_class + 1
    y_fixed = [blank * np.ones(2 * yi.shape[0] + 1).astype('int32')
               for yi in y]
    for i, yi in enumerate(y):
        y_fixed[i][1:-1:2] = yi
    return y_fixed


def np_randn(shape, rng, name=None):
    """ Builds a numpy variable filled with random normal values """
    return (0.01 * rng.randn(*shape)).astype(theano.config.floatX)

def np_rand(shape, rng):
    return (0.01 * (rng.rand(*shape) - 0.5)).astype(theano.config.floatX)


def orthogonal(shape, scale=1.1 ,named=""):
    """ benanne lasagne ortho init (faster than qr approach)"""
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v # pick the one with the correct shape
    q = q.reshape(shape)
    value=scale * q[:shape[0], :shape[1]]
    return shared(value.astype(theano.config.floatX), name=named,borrow=True)

def create_params(shape1,shape2=None,named=""):
    if shape2:
        #values = numpy.asarray(rng.uniform(
        #    low=-numpy.sqrt(6. / (shape1 + shape2)),
        #    high=numpy.sqrt(6. / (shape1 + shape2)),
        #    size=(shape1 + shape2)), dtype=theano.config.floatX)
        #values *= 4
        #return shared(value=values, name=named, borrow=True)
        shape = (shape1, shape2)
        return orthogonal(shape, named)
    else:
        return build_shared_zeros((shape1,), name=named)



########################
#activations. 
import theano
import theano.tensor as T

def softmax(x):
    e_x = T.exp(x - x.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def rectify(x):
    return (x + abs(x)) / 2.0

def tanh(x):
    return T.tanh(x)

def sigmoid(x):
    return T.nnet.sigmoid(x)

def linear(x):
    return x

def t_rectify(x):
    return x * (x > 1)

def t_linear(x):
    return x * (abs(x) > 1)

def maxout(x):
    return T.maximum(x[:, 0::2], x[:, 1::2])

def conv_maxout(x):
    return T.maximum(x[:, 0::2, :, :], x[:, 1::2, :, :])

def clipped_maxout(x):
    return T.clip(T.maximum(x[:, 0::2], x[:, 1::2]), -5., 5.)

def clipped_rectify(x):
    return T.clip((x + abs(x)) / 2.0, 0., 5.)

def hard_tanh(x):
    return T.clip(x, -1. , 1.)

def steeper_sigmoid(x):
    return 1./(1. + T.exp(-3.75 * x))

def hard_sigmoid(x):
    return T.clip(x + 0.5, 0., 1.)



########################
# Layers
########################

class DatasetMiniBatchIterator(object):
    """ Basic mini-batch iterator """
    def __init__(self, x, y, batch_size=200, randomize=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.randomize = randomize
        from sklearn.utils import check_random_state
        self.rng = check_random_state(42)

    def __iter__(self):
        n_samples = self.x.shape[0]
        if self.randomize:
            for _ in xrange(n_samples / BATCH_SIZE):
                if BATCH_SIZE > 1:
                    i = int(self.rng.rand(1) * ((n_samples+BATCH_SIZE-1) / BATCH_SIZE))
                else:
                    i = int(math.floor(self.rng.rand(1) * n_samples))
                yield (i, self.x[i*self.batch_size:(i+1)*self.batch_size],self.y[i*self.batch_size:(i+1)*self.batch_size])
        else:
            for i in xrange((n_samples + self.batch_size - 1)
                            / self.batch_size):
                yield (self.x[i*self.batch_size:(i+1)*self.batch_size],self.y[i*self.batch_size:(i+1)*self.batch_size])




# Pat York
# Courtesy of https://github.com/rakeshvar/rnn_ctc
# With T.eye() removed for k!=0 (not implemented on GPU for k!=0)
class CTCLayer(object):
    def __init__(self, inpt, labels):
        '''
        Recurrent Relation:
        A matrix that specifies allowed transistions in paths.
        At any time, one could
        0) Stay at the same label (diagonal is identity)
        1) Move to the next label (first upper diagonal is identity)
        2) Skip to the next to next label if
            a) next label is blank and
            b) the next to next label is different from the current
            (Second upper diagonal is product of conditons a & b)
        '''
        n_labels = labels[0].shape[0]
        big_I = T.cast(T.eye(n_labels+2), 'float32')
        recurrence_relation = T.cast(T.eye(n_labels), 'float32') + big_I[2:,1:-1] + big_I[2:,:-2] * T.cast((T.arange(n_labels) % 2), 'float32')
        recurrence_relation = T.cast(recurrence_relation, 'float32')

        def step(input, label):
            '''
            Forward path probabilities
            '''
            pred_y = input[:, label]

            probabilities, _ = theano.scan(
                lambda curr, prev: curr * T.dot(prev, recurrence_relation),
                sequences=[pred_y],
                outputs_info=[T.cast(T.eye(n_labels)[0], 'float32')]
            )
            return -T.log(T.sum(probabilities[-1, -2:]))

        probs, _ = theano.scan(
            step,
            sequences=[inpt, labels]
        )

        self.cost = T.cast(T.mean(probs), dtype='float32')
        self.params = []


class Transformation(object):
    """ Basic rectified-linear transformation layer (W.X + b) """
    def __init__(self, rng, input, n_in, n_out, fdrop=True, W=None, b=None, activation=clipped_rectify):
        if W is None:
            W = orthogonal((n_in, n_out), named="W")
        else:
            #assumes that it came from a previous W matrix
            W = theano.shared(value=W, name='W', borrow=True)
        if b is None:
            b = create_params(num_units, named="b")
        else:
            #assumes that it came from a previous b matrix
            b = shared(value=b, name='b', borrow=True)
        self.input = input
        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        self.output = T.dot(self.input, self.W) + self.b
        self.pre_activation = self.output
        if fdrop:
            self.pre_activation = fast_dropout(rng, self.pre_activation)
        self.output = activation(self.pre_activation)
        self.y_pred = T.argmax(self.output, axis=1)

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError("y should have the same shape as self.y_pred",
                ("y", y.type, "y_pred", self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            print("!!! y should be of int type")
            return T.mean(T.neq(self.y_pred, numpy.asarray(y, dtype='int')))

    def __repr__(self):
        return "Transformation"


class BIDIGRU(object):

    def __init__(self, input, n_in=200, n_out=200, n_hidden=200, 
                                truncate_gradient=-1, 
                                seq_output=False,
                                        p_drop=0.,
                                        w_z=None,
                                        w_r=None,
                                        w_h=None,
                                        u_z=None,
                                        u_r=None,
                                        u_h=None,
                                        b_z=None,
                                        b_r=None,
                                        b_h=None,
                                        bw_z=None,
                                        bw_r=None,
                                        bw_h=None,
                                        bu_z=None,
                                        bu_r=None,
                                        bu_h=None,
                                        bb_z=None,
                                        bb_r=None,
                                        bb_h=None,
                                        activation = "tahn"):

        #activation='tanh', gate_activation='steeper_sigmoid',
        self.activation_str = activation
        #self.activation = getattr(activations, activation)
        #self.gate_activation = getattr(activations, gate_activation)
        self.n_hidden = n_hidden
        self.truncate_gradient = truncate_gradient
        self.seq_output = seq_output
        self.p_drop = p_drop
        #self.weights = weights

        self.l_in = input
        self.n_in = n_in


        #forward params
        self.h0 = create_params(self.n_hidden)

        self.w_z = orthogonal((self.n_in, self.n_hidden), named="w_z")
        self.w_r = orthogonal((self.n_in, self.n_hidden), named="w_r")

        self.u_z = orthogonal((self.n_hidden, self.n_hidden), named="u_z")
        self.u_r = orthogonal((self.n_hidden, self.n_hidden), named="u_r")

        self.b_z = create_params(self.n_in, named="b_z")
        self.b_r = create_params(self.n_in, named="b_r")

        if 'maxout' in self.activation_str:
            self.w_h = orthogonal((self.n_in, self.n_hidden*2), named="w_h") 
            self.u_h = orthogonal((self.n_hidden, self.n_hidden*2), named="u_h")
            self.b_h = cr(self.n_hidden*2, named="b_h")
        else:
            self.w_h = orthogonal((self.n_in, self.n_hidden), named="w_h") 
            self.u_h = orthogonal((self.n_hidden, self.n_hidden), named="u_h")
            self.b_h = create_params(self.n_hidden, named="b_h")   


        #backward params
        self.bh0 = create_params(self.n_hidden)

        self.bw_z = orthogonal((self.n_in, self.n_hidden), named="bw_z")
        self.w_r = orthogonal((self.n_in, self.n_hidden), named="bw_r")

        self.bu_z = orthogonal((self.n_hidden, self.n_hidden), named="bu_z")
        self.bu_r = orthogonal((self.n_hidden, self.n_hidden), named="bu_r")

        self.bb_z = create_params(self.n_in, named="bb_z")
        self.bb_r = create_params(self.n_in, named="bb_r")

        if 'maxout' in self.activation_str:
            self.bw_h = orthogonal((self.n_in, self.n_hidden*2), named="bw_h") 
            self.bu_h = orthogonal((self.n_hidden, self.n_hidden*2), named="bu_h")
            self.bb_h = cr(self.n_hidden*2, named="bb_h")
        else:
            self.bw_h = orthogonal((self.n_in, self.n_hidden), named="bw_h") 
            self.bu_h = orthogonal((self.n_hidden, self.n_hidden), named="bu_h")
            self.bb_h = create_params(self.n_hidden, named="bb_h")   

        self.params = [self.h0, self.w_z, self.w_r, self.w_h, self.u_z, self.u_r, self.u_h, self.b_z, self.b_r, self.b_h,
                        self.bh0, self.bw_z, self.bw_r, self.bw_h, self.bu_z, self.bu_r, self.bu_h, self.bb_z, self.bb_r, self.bb_h]



        def step(xz_t, xr_t, xh_t, h_tm1, u_z, u_r, u_h):
            z = steeper_sigmoid(xz_t + T.dot(h_tm1, u_z))
            r = steeper_sigmoid(xr_t + T.dot(h_tm1, u_r))
            h_tilda_t = tanh(xh_t + T.dot(r * h_tm1, u_h))
            h_t = z * h_tm1 + (1 - z) * h_tilda_t
            return h_t

        def forward(dropout_active=False):
            X = self.input
            if self.p_drop > 0. and dropout_active:
                X = dropoutv(X, self.p_drop)
            x_z = T.dot(X, self.w_z) + self.b_z
            x_r = T.dot(X, self.w_r) + self.b_r
            x_h = T.dot(X, self.w_h) + self.b_h
            out, _ = theano.scan(self.step, 
                sequences=[x_z, x_r, x_h], 
                outputs_info=[repeat(self.h0, x_h.shape[1], axis=0)], 
                non_sequences=[self.u_z, self.u_r, self.u_h],
                truncate_gradient=self.truncate_gradient,
                go_backwards = False
            )
            if self.seq_output:
                return out
            else:
                return out[-1]


        def backward(dropout_active=False):
            bX = self.input
            if self.p_drop > 0. and dropout_active:
                bX = dropoutv(bX, self.p_drop)
            bx_z = T.dot(bX, self.bw_z) + self.bb_z
            bx_r = T.dot(bX, self.bw_r) + self.bb_r
            bx_h = T.dot(bX, self.bw_h) + self.bb_h
            out, _ = theano.scan(self.step, 
                sequences=[bx_z, bx_r, bx_h], 
                outputs_info=[repeat(self.bh0, bx_h.shape[1], axis=0)], 
                non_sequences=[self.bu_z, self.bu_r, self.bu_h],
                truncate_gradient=self.truncate_gradient,
                go_backwards = True
            )
            if self.seq_output:
                return out
            else:
                return out[-1]


           self.output = T.concatenate(h_forward(dropout_active=False), h_backward(dropout_active=False), axis=1)



class OCR(object):
	""" Basic OCR module"""
   def __init__(self, numpy_rng, n_in,
   				n_out,
   				theano_rng=None,
                debugprint=False):

        self._rho = 0.95  # "momentum" for adadelta
        self._eps = 1.E-6  # epsilon for adadelta
        self._accugrads = []  # for adadelta
        self._accudeltas = []  # for adadelta

        if theano_rng == None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.x = T.fmatrix('x')
        self.y = T.imatrix('y') #time/space matrix
        self.batch_size = 
        layer_input = self.x

        Rectify1 = Transformation(rng=numpy_rng, 
                        input=layer_input, n_in=n_in, 
                        n_out=2000, 
                        activation = clipped_rectify
                        fdrop = True)
            assert hasattr(this_layer, 'output')
            self.params.extend(this_layer.params)
            self._accugrads.extend([build_shared_zeros(t.shape.eval(),
                'accugrad') for t in this_layer.params])
            self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
                'accudelta') for t in this_layer.params])
            self.layers.append(this_layer)
          

        Rectify2 = Transformation(rng=numpy_rng, 
                        input=Rectify1.output, n_in=2000, 
                        n_out=2000, 
                        activation = clipped_rectify
                        fdrop = True)
            assert hasattr(this_layer, 'output')
            self.params.extend(this_layer.params)
            self._accugrads.extend([build_shared_zeros(t.shape.eval(),
                'accugrad') for t in this_layer.params])
            self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
                'accudelta') for t in this_layer.params])



        Rectify3 = Transformation(rng=numpy_rng, 
                        input=Rectify2.output, n_in=2000, 
                        n_out=2000, 
                        activation = clipped_rectify
                        fdrop = True)
            assert hasattr(this_layer, 'output')
            self.params.extend(this_layer.params)
            self._accugrads.extend([build_shared_zeros(t.shape.eval(),
                'accugrad') for t in this_layer.params])
            self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
                'accudelta') for t in this_layer.params])


        Recurrent1 = BIDIGRU(rng=numpy_rng, 
                        input=Rectify2.output, n_in=2000, 
                        n_out=2000, 
                        activation = clipped_rectify
                        fdrop = True)
            assert hasattr(this_layer, 'output')
            self.params.extend(this_layer.params)
            self._accugrads.extend([build_shared_zeros(t.shape.eval(),
                'accugrad') for t in this_layer.params])
            self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
                'accudelta') for t in this_layer.params])



        Softmax1 = Transformation(rng=numpy_rng, 
                        input=Rectify2.output, n_in=2000, 
                        n_out=n_outs, 
                        activation = softmax
                        fdrop = False)
            assert hasattr(this_layer, 'output')
            self.params.extend(this_layer.params)
            self._accugrads.extend([build_shared_zeros(t.shape.eval(),
                'accugrad') for t in this_layer.params])
            self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
                'accudelta') for t in this_layer.params])



        _CTC = CTCLayer(Softmax1.output, self.y, self.n_out-1)
        self.y_pred = Softmax1.ypred(self.x)
        self.pygivenx = Softmax1.output(self.x)
        self.mean_cost = _CTC.cost
        self.errors = Softmax1.errors(self.y)


    def __repr__(self):
        return "OCRModel"


    def get_adadelta_trainer(self):
        """ Returns an Adadelta (Zeiler 2012) trainer using self._rho and
        self._eps params. """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        gparams = T.grad(self.mean_cost, self.params)
        updates = OrderedDict()
        for accugrad, accudelta, param, gparam in zip(self._accugrads,
                self._accudeltas, self.params, gparams):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = self._rho * accugrad + (1 - self._rho) * gparam * gparam
            dx = - T.sqrt((accudelta + self._eps)
                          / (agrad + self._eps)) * gparam
            updates[accudelta] = (self._rho * accudelta
                                  + (1 - self._rho) * dx * dx)
            updates[param] = param + dx
            updates[accugrad] = agrad

        train_fn = theano.function(inputs=[theano.Param(batch_x),
                                           theano.Param(batch_y)],
                                   outputs=self.mean_cost,
                                   updates=updates,
                                   givens={self.x: batch_x, self.y: batch_y})

        return train_fn



    def score_classif(self, given_set):
        """ Returns functions to get current classification errors. """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        score = theano.function(inputs=[theano.Param(batch_x),
                                        theano.Param(batch_y)],
                                outputs=self.errors,
                                givens={self.x: batch_x, self.y: batch_y})

        def scoref():
            """ returned function that scans the entire set given as input """
            return [score(batch_x, batch_y) for batch_x, batch_y in given_set]

        return scoref


    def predict_(self, given_set):
        batch_x = T.fmatrix('batch_x')
        pred = theano.function(inputs=[theano.Param(batch_x)],
                                outputs=self.y_pred,
                                givens={self.x: batch_x})

        def predict():
            return pred(given_set)
        
        return predict


    def predict_proba_(self, given_set):
        batch_x = T.fmatrix('batch_x')
        pred_prob = theano.function(inputs=[theano.Param(batch_x)],
                                outputs=self.pygivenx,
                                givens={self.x: batch_x})
        def predict_probf():
            return pred_prob(given_set)
        return predict_probf

    #fit automatically trains with adadelta for speed
    def fit(self, x_train, y_train, x_dev=None, y_dev=None,
            max_epochs=300, early_stopping=True, split_ratio=0.1,
            verbose=False, plot=False):

        """
        Fits the neural network to `x_train` and `y_train`. 
        If x_dev nor y_dev are not given, it will do a `split_ratio` cross-
        validation split on `x_train` and `y_train` (for early stopping).
        """
        if x_dev == None or y_dev == None:
            x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train,
                test_size=split_ratio, random_state=42)

        #training function is already adadelta
        train_fn = self.get_adadelta_trainer()

        train_set_iterator = DatasetMiniBatchIterator(x_train, y_train)
        dev_set_iterator = DatasetMiniBatchIterator(x_dev, y_dev)
        train_scoref = self.score_classif(train_set_iterator)
        dev_scoref = self.score_classif(dev_set_iterator)
        best_dev_loss = numpy.inf

        self.method='adadelta'


        patience = 1000  
        patience_increase = 2.  # wait this much longer when a new best is
                                # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant

        done_looping = False
        print '... training the model'
        test_score = 0.
        start_time = time.clock()
        epoch = 0
        timer = None

        if plot:
            verbose = True
            self._costs = []
            self._train_errors = []
            self._dev_errors = []
            self._updates = []

        #if either of these hit len()==3 any time, it stops the training
        self.stopcost = []
        self.stopval= []

        while (epoch < max_epochs) and (not done_looping):
            zeros = []
            trainzeros= []
            epoch += 1
            if not verbose:
                sys.stdout.write("\r%0.2f%%" % (epoch * 100./ max_epochs))
                sys.stdout.flush()
            avg_costs = []
            timer = time.time()
            for iteration, (x, y) in enumerate(train_set_iterator):
                #call the training function
                avg_cost = train_fn(x, y)
                if type(avg_cost) == list:
                    avg_costs.append(avg_cost[0])
                else:
                    avg_costs.append(avg_cost)
            if verbose:
                mean_costs = numpy.mean(avg_costs)
                mean_train_errors = numpy.mean(train_scoref())

                ###hacky stop methods###
                if mean_costs == 0:
                    self.stopcost.append(0)
                    if len(self.stopcost)==3:
                        print 'saving params: will take some time.'
                        #method to extract weight data
                        evals = [x.eval() for x in self.params]

                        zipped = zip(self.params, evals)

                        repo = []
                        for i in range(len(zipped)):
                            if zipped[i][0].name is 'W':
                                pair = zipped[i][1], zipped[i+1][1]
                                repo.append(pair)
                        self.saved_params = repo
                        done_looping = True
                        break
                if mean_train_errors == 0:
                    self.stopval.append(0)
                    if len(self.stopval)==3:
                        print 'saving params: will take some time.'
                        #method to extract weight data
                        evals = [x.eval() for x in self.params]

                        zipped = zip(self.params, evals)

                        repo = []
                        for i in range(len(zipped)):
                            if zipped[i][0].name is 'W':
                                pair = zipped[i][1], zipped[i+1][1]
                                repo.append(pair)
                        self.saved_params = repo
                        done_looping = True
                        break
                #########################

                print('  epoch %i took %f seconds' %
                    (epoch, time.time() - timer))
                print('  epoch %i, avg costs %f' %
                    (epoch, mean_costs))
                print('  epoch %i, training error %f' %
                    (epoch, mean_train_errors))
                if plot:
                    self._costs.append(mean_costs)
                    self._train_errors.append(mean_train_errors)

            dev_errors = numpy.mean(dev_scoref())
            if plot:
                self._dev_errors.append(dev_errors)

            if dev_errors < best_dev_loss:
                best_dev_loss = dev_errors
                best_params = copy.deepcopy(self.params)
                if verbose:
                    print('!!!  epoch %i, validation error of best model %f' %
                        (epoch, dev_errors))
                if (dev_errors < best_dev_loss *
                    improvement_threshold):
                    patience = max(patience, iteration * patience_increase)
                if patience <= iteration:
                    done_looping = True
                    break
                  
        if not verbose:
            print("")
        for i, param in enumerate(best_params):
            self.params[i] = param
 
    def score(self, x, y):
        """ error rates """
        iterator = DatasetMiniBatchIterator(x, y)
        scoref = self.score_classif(iterator)
        return numpy.mean(scoref())


    def plot(self, x_test, y_test,methods=False):
        import matplotlib.pyplot as plt
        plt.figure()
        ax1 = plt.subplot(221)
        ax2 = plt.subplot(222)
        ax3 = plt.subplot(223)
        ax4 = plt.subplot(224)  # TODO updates of the weights
        test_error = self.score(x_test, y_test)
        print("score: %f" % (1. - test_error))
        ax1.plot(numpy.log10(self._costs), label=self.method)
        #ax2.plot(numpy.log10(dnn._train_errors), label=method)
        #ax3.plot(numpy.log10(dnn._dev_errors), label=method)
        ax2.plot(self._train_errors, label=self.method)
        ax3.plot(self._dev_errors, label=self.method)
        #ax4.plot(dnn._updates, label=method) TODO
        ax4.plot([test_error for _ in range(10)], label=self.method)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('cost (log10)')
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('train error')
        ax3.set_xlabel('epoch')
        ax3.set_ylabel('dev error')
        ax4.set_ylabel('test error')
        plt.legend()
        plt.savefig('training_deepspeech.png')

##################################
##################################
##################################
##            Ocropy line models
##################################
##################################
##################################
import sys,os,re
from scipy import stats
from scipy.ndimage import measurements,interpolation,filters
from pylab import *
import common,morph
from toplevel import *
import argparse
from pylab import *



#init from Ocropy models
class Codec:
    """Translate between integer codes and characters."""
    def init(self,charset):
        charset = sorted(list(set(charset)))
        self.code2char = {}
        self.char2code = {}
        for code,char in enumerate(charset):
            self.code2char[code] = char
            self.char2code[char] = code
        return self
    def size(self):
        """The total number of codes (use this for the number of output
        classes when training a classifier."""
        return len(list(self.code2char.keys()))
    def encode(self,s):
        "Encode the string `s` into a code sequence."
        # tab = self.char2code
        dflt = self.char2code["~"]
        return [self.char2code.get(c,dflt) for c in s]
    def decode(self,l):
        "Decode a code sequence into a string."
        s = [self.code2char.get(c,"~") for c in l]
        return s




def scale_to_h(img,target_height,order=1,dtype=dtype('f'),cval=0):
    h,w = img.shape
    scale = target_height*1.0/h
    target_width = int(scale*w)
    output = interpolation.affine_transform(1.0*img,eye(2)/scale,order=order,
                                            output_shape=(target_height,target_width),
                                            mode='constant',cval=cval)
    output = array(output,dtype=dtype)
    return output

class CenterNormalizer:
    def __init__(self,target_height=48,params=(4,1.0,0.3)):
        self.debug = int(os.getenv("debug_center") or "0")
        self.target_height = target_height
        self.range,self.smoothness,self.extra = params
        print "# CenterNormalizer"
    def setHeight(self,target_height):
        self.target_height = target_height
    def measure(self,line):
        h,w = line.shape
        smoothed = filters.gaussian_filter(line,(h*0.5,h*self.smoothness),mode='constant')
        smoothed += 0.001*filters.uniform_filter(smoothed,(h*0.5,w),mode='constant')
        self.shape = (h,w)
        a = argmax(smoothed,axis=0)
        a = filters.gaussian_filter(a,h*self.extra)
        self.center = array(a,'i')
        deltas = abs(arange(h)[:,newaxis]-self.center[newaxis,:])
        self.mad = mean(deltas[line!=0])
        self.r = int(1+self.range*self.mad)
        if self.debug:
            figure("center")
            imshow(line,cmap=cm.gray)
            plot(self.center)
            ginput(1,1000)
    def dewarp(self,img,cval=0,dtype=dtype('f')):
        assert img.shape==self.shape
        h,w = img.shape
        padded = vstack([cval*ones((h,w)),img,cval*ones((h,w))])
        center = self.center+h
        dewarped = [padded[center[i]-self.r:center[i]+self.r,i] for i in range(w)]
        dewarped = array(dewarped,dtype=dtype).T
        return dewarped
    def normalize(self,img,order=1,dtype=dtype('f'),cval=0):
        dewarped = self.dewarp(img,cval=cval,dtype=dtype)
        h,w = dewarped.shape
        # output = zeros(dewarped.shape,dtype)
        scaled = scale_to_h(dewarped,self.target_height,order=order,dtype=dtype,cval=cval)
        return scaled




ascii_labels = [""," ","~"] + [unichr(x) for x in range(33,126)]
digits = u"0123456789"
letters = u"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
symbols = u"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
ascii = digits+letters+symbols
xsymbols = u"""€¢£»«›‹÷©®†‡°∙•◦‣¶§÷¡¿▪▫"""
german = u"ÄäÖöÜüß"
french = u"ÀàÂâÆæÇçÉéÈèÊêËëÎîÏïÔôŒœÙùÛûÜüŸÿ"
turkish = u"ĞğŞşıſ"
greek = u"ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω"
default = ascii+xsymbols+german+french
charset = sorted(list(set(list(ascii_labels) + list(default))))
charset = [""," ","~",]+[c for c in charset if c not in [" ","~"]]
codec = Codec().init(charset)

lnorm = lineest.CenterNormalizer(args.lineheight)












