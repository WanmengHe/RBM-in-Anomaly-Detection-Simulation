from rbm import RBM
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import timeit
import theano.tensor as T
import numpy

# --------------------------------------------------------------------------
class GBRBM(RBM):
    # --------------------------------------------------------------------------
    # initialize class
    # def __init__(self, input, n_in=784, n_hidden=500, \
    # W=None, hbias=None, vbias=None, numpy_rng=None, transpose=False, activation=T.nnet.sigmoid,
    #              theano_rng=None, name='grbm', W_r=None, dropout=0, dropconnect=0):
    def __init__(self, input, n_visible=784, n_hidden=500, W=None, hbias=None, vbias=None, numpy_rng=None,
                 theano_rng=None):
        # initialize parent class (RBM)
        # RBM.__init__(self, input=input, n_visible=n_in, n_hidden=n_hidden, activation=activation,
        #              W=W, hbias=hbias, vbias=vbias, transpose=transpose, numpy_rng=numpy_rng,
        #              theano_rng=theano_rng, name=name, dropout=dropout, dropconnect=dropconnect)
        RBM.__init__(self, input=input, n_visible=n_visible, n_hidden=n_hidden, W=W, hbias=hbias, vbias=vbias,
                     numpy_rng=numpy_rng, theano_rng=theano_rng)

    # --------------------------------------------------------------------------
    def type(self):
        return 'gauss-bernoulli'

    # --------------------------------------------------------------------------
    # overwrite free energy function (here only vbias term is different)
    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = 0.5 * T.dot((v_sample - self.vbias), (v_sample - self.vbias).T)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    # --------------------------------------------------------------------------
    # overwrite sampling function (here you sample from normal distribution)
    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)

        '''
            Since the input data is normalized to unit variance and zero mean, we do not have to sample
            from a normal distribution and pass the pre_sigmoid instead. If this is not the case, we have to sample the
            distribution.
        '''
        # in fact, you don't need to sample from normal distribution here and just use pre_sigmoid activation instead
        # v1_sample = self.theano_rng.normal(size=v1_mean.shape, avg=v1_mean, std=1.0, dtype=theano.config.floatX) + pre_sigmoid_v1
        v1_sample = pre_sigmoid_v1
        return [pre_sigmoid_v1, v1_mean, v1_sample]


def test_gbrbm(learning_rate=0.01, training_epochs=10, batch_size=2, n_hidden=7, n_chains=2, n_samples=10):
    """
    Demonstrate how to train and afterwards sample from it using Theano.

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used for training the RBM

    :param training_epochs: number of epochs used for training

    :param dataset: path the the pickled dataset

    :param batch_size: size of a batch used to train the RBM

    :param n_chains: number of parallel Gibbs chains to be used for sampling

    :param n_samples: number of samples to plot for each chain

    """

    dataset = [[300, 14, 100],
               [311, 12, 110],
               [322, 10, 100],
               [333, 8, 110],
               [344, 6, 100],
               [355, 4, 110]]
    datasets = numpy.array(dataset)
    datasets = datasets.astype(theano.config.floatX)
    train_set_x = datasets
    train_set_x = theano.shared(train_set_x)


    testset = [[1, 2, 3],
               [3, 3, 4],
               [1, 2, 3],
               [2, 3, 4],
               [4, 5, 6],
               [1, 2, 3]]
    testsets = numpy.array(testset)
    testsets = datasets.astype(theano.config.floatX)
    test_set_x = testsets
    test_set_x = theano.shared(test_set_x)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # initialize storage for the persistent chain (state = hidden
    # layer of chain)
    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)

    # construct the RBM class
    gbrbm = GBRBM(input=x, n_visible=3,
                  n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = gbrbm.get_cost_updates(lr=learning_rate,
                                           persistent=persistent_chain, k=15)

    #################################
    # Training the RBM          #
    #################################
    # if not os.path.isdir(output_folder):
    #     os.makedirs(output_folder)
    # os.chdir(output_folder)

    # start-snippet-5
    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        },
        name='train_rbm'
    )

    plotting_time = 0.
    start_time = timeit.default_timer()

    # go through training epochs
    for epoch in xrange(training_epochs):

        # go through the training set
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_rbm(batch_index)]

        print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)

        # Plot filters after each training epoch
        plotting_start = timeit.default_timer()
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start)

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time / 60.))




    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]

    # pick random test examples, with which to initialize the persistent chain
    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(
        numpy.asarray(
            test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
            dtype=theano.config.floatX
        )
    )

    plot_every = 100
    # define one step of Gibbs sampling (mf = mean-field) define a
    # function that does `plot_every` steps before returning the
    # sample for plotting
    (
        [
            presig_hids,
            hid_mfs,
            hid_samples,
            presig_vis,
            vis_mfs,
            vis_samples
        ],
        updates
    ) = theano.scan(
        gbrbm.gibbs_vhv,
        outputs_info=[None, None, None, None, None, persistent_vis_chain],
        n_steps=plot_every
    )

    # add to updates the shared variable that takes care of our persistent
    # chain :.
    updates.update({persistent_vis_chain: vis_samples[-1]})
    # construct the function that implements our persistent chain.
    # we generate the "mean field" activations for plotting and the actual
    # samples for reinitializing the state of our persistent chain
    sample_fn = theano.function(
        [],
        [
            vis_mfs[-1],
            vis_samples[-1]
        ],
        updates=updates,
        name='sample_fn'
    )

    # image_data = numpy.zeros(
    #     (29 * n_samples + 1, 29 * n_chains - 1),
    #     dtype='uint8'
    # )
    for idx in xrange(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        vis_mf, vis_sample = sample_fn()
        print ' ... plotting sample ', idx
        # image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
        #     X=vis_mf,
        #     img_shape=(28, 28),
        #     tile_shape=(1, n_chains),
        #     tile_spacing=(1, 1)
        # )
        # print vis_mf
        print vis_sample


if __name__ == '__main__':
    test_gbrbm()