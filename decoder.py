from lib.gbrbm import GBRBM
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import timeit
import numpy
import theano.tensor as T

def gbrbm_decoder(dataset, learning_rate=0.1, training_epochs=20, batch_size=2, n_hidden=7, plot_every=1):
    dataset = dataset.astype(theano.config.floatX)
    train_set_x = theano.shared(dataset)
    n_visible = dataset.shape[1]

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
    gbrbm = GBRBM(input=x, n_visible=n_visible, n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = gbrbm.get_cost_updates(lr=learning_rate,
                                           persistent=persistent_chain, k=15)

    #################################
    # Training the RBM          #
    #################################
    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
            # x: train_set_x[index: index + batch_size]
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

    persistent_vis_chain = theano.shared(
        numpy.asarray(
            train_set_x.get_value(borrow=True),  # [test_idx:test_idx + n_chains],
            dtype=theano.config.floatX
        )
    )

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

    # add to updates the shared variable that takes care of our persistent chain :.
    updates.update({persistent_vis_chain: vis_samples[-1]})
    # construct the function that implements our persistent chain.
    # we generate the "mean field" activations for plotting and the actual
    # samples for reinitializing the state of our persistent chain
    sample_fn = theano.function(
        [],
        [
            vis_mfs[-1],
            vis_samples[-1],
            hid_samples[-1]
        ],
        updates=updates,
        name='sample_fn'
    )

    vis_mf, vis_sample, hid_sample = sample_fn()
    return vis_mf, vis_sample, hid_sample


if __name__ == '__main__':
    datasets = numpy.array([[0, 0, 0],
                            [0, 0, 0],
                            [322, 10, 100],
                            [333, 8, 110],
                            [344, 6, 100],
                            [0, 0, 0]])
    gbrbm_decoder(datasets)
