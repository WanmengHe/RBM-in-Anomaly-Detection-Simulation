Decoding Feature Vectors of Network Packets
===

What's this?
---
This project intends to implement GBRBM & RNN-RBM to decode a feature vector of network packets into a binary array, in order to discriminate the anomaly packets in network traffic.
It mainly focus on converting a real vector to a binary vector, which keep those vectors which contain anomaly values away from the normal vectors in the binary vector space.

The following two pictures indicate that there are more distinct discrimination between anomaly data(blue), and normal data. The pictures are the distribution of data points which were processed by T-SNE which is used to reduce the dimensionality of data.
![generated data after T-SNE](https://github.com/meowoodie/rbm-in-anomaly-detection-simulation/blob/master/data/N6_n1000_t5_e1_gbrbm_h500/generated_data_3D_scatter.png)
![decoded data after T-SNE](https://github.com/meowoodie/rbm-in-anomaly-detection-simulation/blob/master/data/N6_n1000_t5_e1_gbrbm_h500/decoded_data_3D_scatter.png)

How to use this?
---
I have implemented a experiment component to run a variety of different experiments, also you can implement your experiment in experiment.py.
```python
# use gbrbm to decode a dataset which contains 1000 tuples
# and five of them are anomaly data points.
exp_gbrbm("N6_n1000_t5_e1_gbrbm_h500", T=[0, 1, 2, 3, 4, 5])
```
The result would be generated at the directory `data/N6_n1000_t5_e1_gbrbm_h500/`. The results include data file and pictures.

