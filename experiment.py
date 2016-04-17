import utils
import decoder
import data_generator as dg

dataset = dg.one_exception_dataset(
    N=6,
    n=1000,
    T=[0, 1, 2],
    lam=5000,
    exc=1,
    noise_k=0
)
utils.write_data(dataset, "one_exception_N6_n1000_t1")
_, recovery_sample, decode_res = decoder.gbrbm_decoder(
    dataset,
    learning_rate=0.1,
    training_epochs=30,
    batch_size=1001,
    n_hidden=20,
    plot_every=1
)
utils.write_data(decode_res, "decode_one_exception_N6_n1000_t1")

dataset = utils.read_data("one_exception_N6_n1000_t1")
# utils.plot_dataset(dataset)