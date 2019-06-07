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
## 1. Introduction
In Neural Network Language Models(NNLM) with huge number of words in vocabulary, exhaustive activation functions such as Softmax are very slow.  This paper addresses shortcomings of Softmax. It consists of mainly two ideas
1. Representing words as low-dimensional feature vectors - to learn relation between words and contexts.
2. Clustering similar words in similar components(subtree) using the feature vectors.

Following is the summary of the Hierarchical Log-Bilinear Model. (If this explanation doesn't summarise the content please go to Section 4 in the Paper)
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
        * Each of q<sub>i</sub> are multiplied with output r_hat and activated using sigmoid. Gives the probability of decision going to left subtree. <p align='center'> P(d<sub>i</sub> = 1): sigmoid( r_hat * q<sub>i</sub> + b<sub>i</sub>) </p>
        * Each leaf is scored according to it's decision code. For example: 
          * leaf_5: P(d<sub>1</sub>=1, d<sub>2</sub>=1, d<sub>3</sub>=1)
          * leaf_3: P(d<sub>1</sub>=0, d<sub>3</sub>=1, d<sub>6</sub>=1)
<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/tree.png'>
</p>

## 2. Architecture

Given a vocabulary of words **V**, 

Number of words/leaves will be **|V|**.

The number of internal nodes will be equal to **|V|-1**

Input **r_hat** given to this function is an array of Word Feature Vector. In our case: (1, 100)
<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/r_hat.png' width=810 height=50 /> 
</p>
Equation 2 is executed as described in following steps:
<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/sigmoid.png' width=310>
</p>

* Step-1:
Here **q**<sub>i</sub> is a vector of shape (100, 1). Hence, a matrix of shape **node_vector** = (**|V|-1**, 100, 1) is created for **q**<sub>i</sub> for i = (1, **|V|-1**)

<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/node_vectors.png' width=810 /> 
</p>

* Step-2: 
Each **q**<sub>i</sub> x **r_hat**, which is: **node_vector** x **r_hat**

This produces a (**|V|-1**, 1) matrix

<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/intermed_q.png' width=510 /> 
</p>

* Step-3: 
d<sub>i</sub> = sigmoid(**node_vector**) produces a (**|V|-1**, 1) matrix, each value consisting of the probability of choosing left child.

<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/left_child_probs.png' width=510 /> 
</p>

* Step-4: 
Each word/leaf has a path from the root. Length of this path will be the height of the tree.
<p align='center'> <b>h</b> = Ceiling(lg(<b>|V|</b>)) </p>

Hence each **|V|** is defined by a set of **h** internal nodes. **|V|** leafs will have less than or equal to **h** nodes in the path from root to leaf. Since, number of nodes in path will be a subset of d<sub>i</sub>, a sparse matrix **decision_matrix** of shape: [**|V|**, **|V|-1**] is created with each node in path consisting of 1 for left child and -1 for right child:
<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/Decision_matrix.png' width=270>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/matrix1.jpg' width=270>
</p>

* Step-5:
Row wise multiplication: **intermed_path_prob** = **d**<sub>i</sub> x **decision_matrix** produces [**|V|**, **|V|-1**] with matrix respective node probabilities. **base** is a base matrix with value 1 in the location where node in **mat2** is negative(right child) and 0 for positive(left child).
<p align='center'>
<b>corrected_probs</b> = <b>base</b> + <b>intermed_path_prob</b>

<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/base_prob.png' width=270>
  =
 <img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/base.png' width=270>
  +
 <img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/intermed_path_probs.png' width=270>
</p>

Step 4 and 5 calculate probabilities of respective nodes: **P(right_child)** from **(1-P(left_child))** and **P(node not in path) = 0**

This logic is explained in: [Hierarchical Softmax as output](https://becominghuman.ai/hierarchical-softmax-as-output-activation-function-in-neural-network-1d19089c4f49)

### 2.1. Final Leaf probability calculation
We do this in either of two ways: 
#### 2.1.1. Directly Multiplying node probabilities. 
reduce_prod function from tensorflow multiplies all the node probabilities of d<sub>i</sub> of each row(leaf or word).
This method gives a constant computation time of **O(lg|V|)**. This operation reduces **corrected_probs** (shape: [**|V|**, **|V|-1**]) to the final probabilities (shape: [**|V|**, 1]).
<p align='center'> 
  <b> final_prob </b> = reduce_product(<b>corrected_probs</b>)
  </p>
<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/Final_probs.png' width=270>
</p>

### 2.1.2. Reducing matrix and using Log Method

<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/Time-%20Log%20method%20vs%20Reduce%20Product.png' width=300>
</p>
## 3. Results
