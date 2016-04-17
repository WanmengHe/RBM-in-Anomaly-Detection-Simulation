Decoding Feature Vectors of Network Packets
===

What's this?
---
This project intends to implement GBRBM & RNN-RBM to decode a feature vector of network packets into a binary array, in order to discriminate the anomaly packets in network traffic.
It mainly focus on converting a real vector to a binary vector, which keep those vectors which contain anomaly values away from the normal vectors in the binary vector space.

The following two pictures indicate that there are more distinct discrimination between anomaly data(blue), and normal data. The pictures are the distribution of data points which were processed by T-SNE which is used to reduce the dimensionality of data.

![generated data after T-SNE in 3D](https://github.com/meowoodie/rbm-in-anomaly-detection-simulation/blob/master/data/N6_n1000_t5_e1_gbrbm_h500/generated_data_3D_scatter.png)
![decoded data after T-SNE in 3D](https://github.com/meowoodie/rbm-in-anomaly-detection-simulation/blob/master/data/N6_n1000_t5_e1_gbrbm_h500/decoded_data_3D_scatter.png)

How to use this?
---
I have implemented a experiment component to run a variety of different experiments, also you can implement your experiment in experiment.py.
```python
# use gbrbm to decode a dataset which contains 1000 tuples
# and five of them are anomaly data points.
exp_gbrbm("N6_n1000_t5_e1_gbrbm_h500", T=[0, 1, 2, 3, 4, 5])
```
The result would be generated at the directory `data/N6_n1000_t5_e1_gbrbm_h500/`. The results include data file and pictures.

Results
---
I have tried a lot of combination of parameters and I have got a wonderful result when I generated a 1500 points dataset and there were 5 anomaly points among them.
Here is the [original feature vectors][] which are a bunch of 6-dimensional real vectors, and here is the [decoded binary vectors][] which are the same number as original feature vectors, and are 2000-dimension.
I used GBRBM to decode the 1500 points from 6-dimensional real vectors to 2000-dimensional binary vectors, and they were visualized by T-SNE.
It was surprised that the 2000-dimensional binary vectors has a distinct border between normal data and anomaly data in the perspective of 2D space.

- 6-dimensional real vectors (origin points) in 2D space.
![generated data after T-SNE in 2D](https://github.com/meowoodie/rbm-in-anomaly-detection-simulation/blob/master/data/N6_n1500_t5_e1_gbrbm_h2000_2D/generated_data_2D_scatter.png)

- 2000-dimensional binary vectors (decoded points) in 2D space.
![decoded data after T-SNE in 2D](https://github.com/meowoodie/rbm-in-anomaly-detection-simulation/blob/master/data/N6_n1500_t5_e1_gbrbm_h2000_2D/decoded_data_2D_scatter.png)

[original feature vectors]:https://github.com/meowoodie/rbm-in-anomaly-detection-simulation/blob/master/data/N6_n1500_t5_e1_gbrbm_h2000_2D/generated_data.txt
[decoded binary vectors]:https://github.com/meowoodie/rbm-in-anomaly-detection-simulation/blob/master/data/N6_n1500_t5_e1_gbrbm_h2000_2D/decoded_data.txt
