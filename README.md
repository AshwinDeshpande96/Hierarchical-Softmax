# Hierarchical-Softmax
This is a scalable hierarchical softmax layer for Neural Networks with large output classes.
In our previous project Next-Word Prediction: [Next-Word Prediction](https://github.com/AshwinDeshpande96/Speech-Generation)
there was an issue of large vocabulary. There was a bottleneck at the Softmax Layer due to the large number of output classes.
In order to solve this issue many solutions were proposed:
  1. Hierarchical Softmax
  2. Differentiated Softmax
  3. Target Sampling
  4. Noise Contrastive Estimation
  5. Infrequent Normalization

You can read about it here: [Strategies for Training Large Vocabulary Neural Language Models](https://arxiv.org/abs/1512.04906)

This project is mainly an implementation of Geoff Hinton's paper: [A Scalable Hierarchical Distributed Language Model](https://www.cs.toronto.edu/~amnih/papers/hlbl_final.pdf).

In Neural Network Language Models(NNLM) with huge number of words in vocabulary, exhaustive activation functions such as Softmax are very slow.  This paper addresses shortcomings of Softmax. This paper consists of mainly two ideas
1. Representing words as low-dimensional feature vectors - so that words with similar meaning and usage have similar feature vectors
2. Clustering similar words in similar components(subtree) using the feature vectors.

In this paper Prof. Hinton talks about a clustering algorithm. I will define it as follows. (If this explanation doesn't summarise the content please go to Section 4 in the Paper)
* Initially start with a random binary tree. With words as leaf.
* Use log-bilinear model to fit training data. 
  * Input will be context: w<sub>1</sub>,w<sub>2</sub>,...,w<sub>n-1</sub>. 
    * Each word w is represented by a feature vector r<sub>w<sub>1</sub></sub>. Say shape (1,100) each.
    * So input at each forward pass will be (n-1, 1, 100)
  * Hidden Layers apply matrix transformations, with weights C
<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/Screenshot%202019-06-05%20at%208.38.05%20PM.png' width=150>
</p>
  
  * Output will be w<sub>n</sub>
    * Will be a predicted feature vector r_hat
    * So output shape at each forward pass will be (1,100)
      * If there are 8 words in vocabulary (output classes) - Fig 1
        * Each of q<sub>i</sub> are multiplied and activated using sigmoid. Gives the probability of decision going to left subtree. <p align='center'> P(d<sub>i</sub> = 1): sigmoid( r_hat * q<sub>i</sub> + b<sub>i</sub>) </p>
        * Each leaf is scored according to it's decision code. For example: 
          * leaf_5: P(d<sub>1</sub>=1, d<sub>2</sub>=1, d<sub>3</sub>=1)
          * leaf_3: P(d<sub>1</sub>=0, d<sub>3</sub>=1, d<sub>6</sub>=1)
<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/tree.png'>
</p>
