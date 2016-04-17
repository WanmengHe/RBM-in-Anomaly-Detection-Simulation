import utils
import decoder
import data_generator as dg

def exp_gbrbm(exp_name, T=[]):
    dataset = dg.one_exception_dataset(
        N=6,
        n=1000,
        T=T,
        lam=5000,
        exc=1,
        noise_k=0
    )
    utils.write_data(dataset, exp_name, "generated_data")

    _, recovery_sample, decode_res = decoder.gbrbm_decoder(
        dataset,
        learning_rate=0.1,
        training_epochs=30,
        batch_size=1001,
        n_hidden=200,
        plot_every=1
    )
    utils.write_data(decode_res, exp_name, "decoded_data")

    new_data = utils.tsne(decode_res, exp_name, T, 3)
    utils.write_data(new_data, exp_name, "tsne_data")

if __name__ == "__main__":
    exp_gbrbm("N6_n1000_e3_gbrbm_h200", T=[0, 1, 2])
