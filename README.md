# Hierarchical-Softmax
This is a scalable hierarchical softmax layer for Neural Networks with large output classes.
In our previous project Next-Word Prediction: [Next-Word Prediction](https://github.com/AshwinDeshpande96/Speech-Generation)
there was an issue of large vocabulary. There was a bottleneck at the Softmax Layer due to the large number of output classes.
In the paper [Strategies for Training Large Vocabulary Neural Language Models](https://arxiv.org/abs/1512.04906) many solutions are proposed:
  1. Hierarchical Softmax
  2. Differentiated Softmax
  3. Target Sampling
  4. Noise Contrastive Estimation
  5. Infrequent Normalization

This project builds on the idea in Geoff Hinton's paper: [A Scalable Hierarchical Distributed Language Model](https://www.cs.toronto.edu/~amnih/papers/hlbl_final.pdf).
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
      * If there are 8 words in vocabulary (output classes)(Fig-1)
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

Input **r_hat** given to this function is an array of Word Feature Vector. In our case the vector is of shape (1,100) (Fig-2)
<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/r_hat.png' width=650 height=100 /> 
</p>
Eq-2 is executed as described in following steps:
<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/sigmoid.png' width=310>
</p>

* Step-1: Here **q**<sub>i</sub> is a vector of shape (100, 1). Hence, a matrix of shape **node_vector** = (**|V|**-1, 100, 1)(Fig-3) is created for **q**<sub>i</sub> where i = 1 to  **|V|**-1.

<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/node_vectors.png' width=550 /> 
</p>

* Step-2: 
Next Operation: **q**<sub>i</sub> x **r_hat** for all i = 1 to **|V|**-1. 

**node_vector** consists of **q**<sub>i</sub> for all i = 1 to **|V|**-1.

**node_vector** x **r_hat** produces a (**|V|-1**, 1) intermediate matrix (Fig-4)

<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/intermed_q.png' width=510 /> 
</p>

* Step-3: Next operation consists of producing the P(d<sub>i</sub>=1) - Probability of decision from node q<sub>i</sub> to take left child route.
**d**<sub>i</sub> = sigmoid(**node_vector**) produces a (**|V|-1**, 1) matrix(Fig-5), each value consisting of the probability of choosing left child.

<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/left_child_probs.png' width=510 /> 
</p>

* Step-4: Previous step produces probability of the decision to take left-child decision in every node in the tree. But the path from root to one particular leaf will consist of only subset of all **|V|**-1 nodes. Length of this path will be less than or equal to height **h** of the tree. 
<p align='center'> <b>h</b> = Ceiling(lg(<b>|V|</b>)) </p>

Hence a matrix is created with **|V|** rows, and each row consists **|V|**-1 columns. Column values is either of three values values (1, -1, 0) signifying left-child, right-child or not-in-path. This matrix is called **decision_matrix**. This is a sparse matrix of shape: (**|V|**, **|V|-1**). (Fig-6)
<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/Decision_matrix.png' width=270>
</p>

* Step-5: Since we have decisions for every leaf, we can now obtain the probabilities associated with those decisions. 

1. First step is to negate P(d_i=1)(Left-decision probabilities) in order to get '-P(d_i = 1)' part of P(d_i = 0) = (1 - P(d_i = 1)) => (Right-decision probabilities). This produced by row-wise multiplication of left-child probabilities and decision matrix.
<p align='center'><b>intermed_path_prob</b> = <b>d**</b><sub>i</sub> x <b>decision_matrix</b> </p> 

This produces a (**|V|**, **|V|-1**) matrix - **intermed_path_prob**(Fig-7(c)). 

2. Second step is to obtain 1 P(d_i=0) = 1 + (-P(d_i = 1)) and 0 in P(d_i=0) = 0 + P(d_i=1).

For this purpose **base** is a matrix with value 1 in the location where node in **intermed_path_prob** is negative(right child) and 0 for positive(left child). (Fig-7(b)) 

`Note: base will consist of 1 in places where node is not present in path of that path. This does not mean that node has 100% probability, it is a minor correction to obtain the multplication of d_i(Eq-2) along say: leaf_x: [0.1, 0.2, 0.3, 0, 0, 0]. this will obtain a P(leaf_x) = 0, hence it converted to [0.1, 0.2, 0.3, 1, 1, 1] to obtain P(leaf_x) = 0.0006`
<p align='center'>
  <b>corrected_probs</b> = <b>base</b> + <b>intermed_path_prob</b>
  </p>
<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/base_prob.png' width=270>
 <img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/base.png' width=270>
 <img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/intermed_path_probs.png' width=270>
</p>

For further details: [Hierarchical Softmax as output activation function in neural-network](https://becominghuman.ai/hierarchical-softmax-as-output-activation-function-in-neural-network-1d19089c4f49)

### 2.1. Final Probability
We do this in either of two ways: 
#### 2.1.1. Reduce Product
reduce_prod function from tensorflow multiplies all the node probabilities of d<sub>i</sub> of each row(leaf or word).
This method gives a constant computation time of **O(lg|V|)**. This operation reduces **corrected_probs** (shape: (**|V|**, **|V|-1**)) to the final probabilities (shape: (**|V|**, 1)).
<p align='center'> 
  <b> final_prob </b> = reduce_product(<b>corrected_probs</b>)
  </p>
<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/Final_probs.png' width=510>
</p>

#### 2.1.2. Log Method
This method was proposed to minimize repetitive into one's. For example the decisions probabilities for left most leaf of a 10,000 size vocabulary will be **D** = [p<sub>1</sub>, p<sub>2</sub>, ..., p<sub>14</sub>, 1<sub>15</sub>, 1<sub>16</sub>, 1<sub>17</sub>, .... ,1<sub>9999</sub>].
We obtain p<sub>1</sub> x p<sub>2</sub> x p<sub>3</sub> x ... x p<sub>14</sub> in two step process.
1. log<sub>n</sub>(**D**) to obtain log<sub>n</sub>(p<sub>1</sub>) , log<sub>n</sub>(p<sub>2</sub>) , log<sub>n</sub>(p<sub>3</sub>) , .... , log<sub>n</sub>(p<sub>14</sub>) ,  log<sub>n</sub>(1<sub>15</sub>) , log<sub>n</sub>(1<sub>16</sub>) , log<sub>n</sub>(1<sub>17</sub>) , .... , log<sub>n</sub>(1<sub>9999</sub>)
<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/Time-%20Log%20method%20vs%20Reduce%20Product.png' width=300>
</p>
## 3. Results
