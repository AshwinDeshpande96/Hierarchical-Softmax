# Hierarchical-Softmax
This is a scalable hierarchical softmax layer for Neural Networks with large output classes.
In my previous project Next-Word Prediction: https://github.com/AshwinDeshpande96/Speech-Generation
there was an issue of large vocabulary. There was a bottleneck at the Softmax Layer due to the large number of output classes.
In order to solve this issue many solutions were proposed:
  1. Hierarchical Softmax
  2. Differentiated Softmax
  3. Target Sampling
  4. Noise Contrastive Estimation
  5. Infrequent Normalization

You can read about it here: https://arxiv.org/abs/1512.04906
