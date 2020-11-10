# Hierarchical-Softmax
This is a scalable hierarchical softmax layer for Neural Networks with large output classes.
In our previous project Next-Word Prediction: [Next-Word Prediction](https://github.com/AshwinDeshpande96/Speech-Generation)
there was an issue of large vocabulary. There was a bottleneck at the Softmax Layer due to the large number of output classes.
Since softmax is an exhaustive method of calculating probabilities distribution across the output classes, it scales poorly with growing size of the vocabulary. Softmax needs a vector that produces scores for each class. This is done only to facilitate the Softmax method. That is, the vector need not be as long as the size of vocabulary. In order to obtain this size, even a smaller feature vector(vector which represents the word's context - these are normally of size 256 or less) is scaled up to meet softmax requirements. A huge amount of parameters are created in the final fully connected layer. They are usually of the scale (feature_vector_size * vocabulary_size). For example, a vocabulary of size only 5000 will need 256 * 5000 parameters, i.e. 1280000 parameters only a single layer. This usually makes up for more than half of the total parameters
Given that feature vector is already obtained in the previous layer we needn't scale it up to another vector. In order to solve this issue we employed methods described in this project.


In the paper [Strategies for Training Large Vocabulary Neural Language Models](https://arxiv.org/abs/1512.04906) few solutions are proposed:
  1. Hierarchical Softmax
  2. Differentiated Softmax
  3. Target Sampling
  4. Noise Contrastive Estimation
  5. Infrequent Normalization

Hierarchial Softmax has great scalibility features. This is not a softmax method, i.e. it does not produce normalized exponentiated probabilities. We will see the proposed method as follows:

This project builds on the idea in Geoff Hinton's paper: [A Scalable Hierarchical Distributed Language Model](https://www.cs.toronto.edu/~amnih/papers/hlbl_final.pdf) - [Lecture](https://www.youtube.com/watch?v=Rtk_juucCHc)

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
        * Each of q<sub>i</sub> are multiplied with output r_hat and activated using sigmoid. Gives the probability of decision going to left subtree. 
        <p align='center'> P(d<sub>i</sub> = 1): sigmoid( r_hat * q<sub>i</sub> + b<sub>i</sub>) </p>
        
        * Each leaf is scored according to it's decision code. For example: 
          * leaf_5: P(d<sub>1</sub>=0, d<sub>3</sub>=1, d<sub>6</sub>=1)
          * leaf_3: P(d<sub>1</sub>=1, d<sub>2</sub>=0, d<sub>5</sub>=1)
* Fit the model with given data
  * This is a teacher-forcing type of model, output at time step t is used at the next step t+1.
  * This creates feature vectors r_hat depending on the context as we train.
<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/tree.png'>
</p>

Task specific feature vector perform well in every NLP task, because the tailored feature vector represent the training data well: [Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)



## 2. Architecture

We devise a method where the control statements involved in fetching subset of nodes from a hierarchical binary tree are converted into direct matrix multiplication. This is done because in libraries such as keras the operations in forward-function needs to be differentiable. Hence, all the operations are matrix manipulation type that have gradients defined in keras which allows for back-propagation.

The method is as follows:

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

`Note: We do not include biases, but incorporating bias is a trivial task: Adding bias will not make any changes to shape of the vectors. If necessary can be added after Step-2.
`

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
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/matrix_decisions.gif' width=560>
</p>

* Step-5: Since we have decisions for every leaf, we can now obtain the probabilities associated with those decisions. 

1. First step is to negate P(d_i=1)(Left-decision probabilities) in order to get '-P(d_i = 1)' part of P(d_i = 0) = (1 - P(d_i = 1)) => (Right-decision probabilities). This produced by row-wise multiplication of left-child probabilities and decision matrix.
<p align='center'><b>intermed_path_prob</b> = <b>d</b><sub>i</sub> x <b>decision_matrix</b> </p> 

This produces a (**|V|**, **|V|-1**) matrix - **intermed_path_prob**(Fig-7(c)). 

2. Second step is to obtain 1 P(d_i=0) = 1 + (-P(d_i = 1)) and 0 in P(d_i=0) = 0 + P(d_i=1).

For this purpose **base** is a matrix with value 1 in the location where node in **intermed_path_prob** is negative(right child) and 0 for positive(left child). (Fig-7(b)) 

`Note: base will consist of 1 in places where node is not present in path of that path. This does not mean that node has 100% probability, it is a minor correction to obtain the multplication of d_i(Eq-2) along say: leaf_x: [0.1, 0.2, 0.3, 0, 0, 0]. this will obtain a P(leaf_x) = 0, hence it converted to [0.1, 0.2, 0.3, 1, 1, 1] to obtain P(leaf_x) = 0.0006`
<p align='center'>
  <b>corrected_probs</b> = <b>base</b> + <b>intermed_path_prob</b>
</p>

<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/base_prob.png' width=260, height=300>
 <img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/base.png' width=300, height=290>
 <img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/intermed_path_probs.png' width=280>
</p>

For further details: [Hierarchical Softmax as output activation function in neural-network](https://becominghuman.ai/hierarchical-softmax-as-output-activation-function-in-neural-network-1d19089c4f49)

### 2.1. Final Probability

We do this in either of two ways: 
#### 2.1.1. Reduce Product

reduce_prod function from tensorflow multiplies all the node probabilities of d<sub>i</sub> of each row(leaf or word).
This method gives a constant computation time of **O(lg|V|)**. This operation reduces **corrected_probs** (shape: (**|V|**, **|V|-1**)) to the final probabilities (shape: (**|V|**, 1)).(Fig-8)
        
    1 def hierarchical_softmax2(inp, tree):
    2   x1 = tf.multiply(tree.decision_matrix, inp)
    3   x1 = tree.base + x1
    4   return tf.math.reduce_prod(x1, axis=1)
    
 <p align='center'> Code-1: Reduce Product Hierarchical Softmax Function      </p>
      
<p align='center'> 
  <b> final_prob </b> = reduce_product(<b>corrected_probs</b>)
  </p>

#### 2.1.2. Log Method
Taking idea of negative log-likelihood wherein multiplication is replaced with summation, because multiplication incurs more comptutation costs than sum, this method was proposed to minimize repetitive multiply by one. 

For example the decisions probabilities for left most leaf of a 10,000 size vocabulary will be **D** = [p<sub>1</sub>, p<sub>2</sub>, ..., p<sub>14</sub>, 1<sub>15</sub>, 1<sub>16</sub>, 1<sub>17</sub>, .... ,1<sub>9999</sub>].
We obtain p<sub>1</sub> x p<sub>2</sub> x p<sub>3</sub> x ... x p<sub>14</sub> in three step process.

1. Logarithm of **corrected_probs**: First Row(**D**) of that operation: <p align='center'>
  log<sub>n</sub>(<b>D</b>) to obtain log<sub>n</sub>(p<sub>1</sub>) , log<sub>n</sub>(p<sub>2</sub>) , log<sub>n</sub>(p<sub>3</sub>) , .... , log<sub>n</sub>(p<sub>14</sub>) ,  log<sub>n</sub>(1<sub>15</sub>) , log<sub>n</sub>(1<sub>16</sub>) , log<sub>n</sub>(1<sub>17</sub>) , .... , log<sub>n</sub>(1<sub>9999</sub>) </b>
</p>

2. Summation of logged values: <p align='center'> Sum(log<sub>n</sub>(p<sub>1</sub>) , log<sub>n</sub>(p<sub>2</sub>) , log<sub>n</sub>(p<sub>3</sub>) , .... , log<sub>n</sub>(p<sub>14</sub>) ,  log<sub>n</sub>(1<sub>15</sub>) , log<sub>n</sub>(1<sub>16</sub>) , log<sub>n</sub>(1<sub>17</sub>) , .... , log<sub>n</sub>(1<sub>9999</sub>) obtains </p> <p align='right'> = log<sub>n</sub>(p<sub>1</sub> x p<sub>2</sub> x p<sub>3</sub> x ... x p<sub>14</sub>)</p>

3.  Exponentiate value to obtain the product:

<p align='center'> p<sub>1</sub> x p<sub>2</sub> x p<sub>3</sub> x ... x p<sub>14</sub> = e<sup>log<sub>n</sub>(p<sub>1</sub> x p<sub>2</sub> x p<sub>3</sub> x ... x p<sub>14</sub>)</sup>
</p>

Both methods result in the same probability (Fig-8), but log method has a disadvantage of losing information in logarithm and exponent function. Even though the probability prediction by both method is correct (Verification: Sum all values to obtain 1.0 - 100% Probability accross all possible output classes), floating point precision is reduced to limited decimal places (rounded off) in the log and exp operation as compared to Reduce Product method which keeps all less important decimal digits.
<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/Final_probs.png' width=510>
</p>

But we see that by inducing the three step process(line-4 and line-6) the computational cost increases. (Fig-9)

    1 def hierarchical_softmax(inp, tree):
    2   x1 = tf.multiply(tree.decision_matrix, input)
    3   x1 = tree.base + x1
    4   x1 = tf.log(x1)                                   #extra step #1
    5   x1 = tf.math.reduce_sum(x1, axis=1)               #reduce_prod is replaced by reduce_sum
    6   return tf.math.exp(x1)                            #extra step #2
    
<p align='center'> Code-2: Log Method Hierarchical Softmax Function </p>
          
<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/Time-%20Log%20method%20vs%20Reduce%20Product.png' width=400>
</p>

## 3. Results

In order to test scalability we do not integrate Hierarchical-Softmax algorithm into a language model. Since probability distribution is calculated at the end of a neural network. We need only test the computational cost that is incurred in the output layer. The layers preceding the output layer of a language model, incur same delay for either Softmax or Hierarchical Softmax. Time taken to calculate probability distribution among the |V| classes remain independent of the predicted feature vector-r_hat, given the size of the feature vector remains unchanged. 

Hence we simulate a condition where a feature vector of  shape (1, |V|-1) is randomly generated every iteration.
* Simulated Word Vector r_hat is generated once for each Vocabulary Size.
  * We chose vocabulary sizes: [1000, 5000, 10000, 15000, 16000], increasing incrementally
  * 16000-18000 is the asymptotic limit for memory of 12GB. This algorithm is limited to a vocabulary of size 18,000 due to the usage decision matrix. Decision matrix consumes memory of the size |V| * (|V|-1), which is in quadratic order:
    * Decision Matrix Method: O(|V|<sup>2</sup>)
    * Softmax Method: O(|V|)
 
 As significant as the speed-ups are it is limited to available memory, hence optimum solution will be a trade-off. Availability of every decision is a major catalyst in estimating node probabilities. Depending on available memory we can partially use decision matrix and partially calculate the decision paths at runtime.
* A vector of shape (1, |V|-1) is generated 5 times each iteration and used for both algorithms sequentially.

Initial vocabulary size is 1000 and not lower as the performance of hierarchical structure is best evaluated for larger data sizes. While this algorithms performs well at lower sizes, it's scalability is best judged when the data size is increased dramatically. When the vocabulary size increases from 5k to 10k, i.e. it is double the time taken remains almost constant. This is due to the nature of the logarithmic asymptote, where the time taken may increase at lower vocabulary sizes, but plateaus eventually.

We see significant difference in the computational cost between the softmax and hierarchical softmax model.
Following is the asymptotic relation with respected to increasing vocabulary size **|V|**.
* Softmax: O(|V|) - (Code-3)
* Hierarchical Softmax: O(lg|V|) - (Code - 1)

      def Softmax(input):
        e = tf.math.exp(input)
        s = tf.math.reduce_sum(e)
        return e/s
        
<p align='center'> Code-3: Softmax Function </p>
        
This is reflected very closely in run-time measurements. From Fig-10 we can see that Hierarchical-Softmax time remains almost constant while Softmax time increases linearly.
<p align='center'>
<img src='https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/Time-%20Softmax%20vs%20Hierarchical%20Softmax.png' width=460>
</p>



# Conclusion & Future Work

Hierarchial Softmax has been proven to reduce computation time, although at the cost of some accuracy the speed-up is substantial. Our matrix method of implementation contributes easy to use and efficient interface to hierarchical softmax architecture. The **decision_matrix** and **base** are created when the model is built and not while training/testing, therefore is majority of computation is performed only once, irrespective of the EPOCH length. Command: `tree = Tree(|V|)` consists of everything from creating trees, paths, decision, decision_matrix & base. It is further easily accessible by simply using `tree` object: tree.decision & tree.base returns computed matrices.

In future, we shall try to reduce memory costs to an order of O(|V|*log|V|) and integrate this method in keras library for easy implementation. User shall be able to use this algorithm as they would with traditional Softmax algorithm.

*An investigation of negative sampling in contrast with hierarchical softmax will be published soon.*

Full Code: [Hierarchical Softmax](https://github.com/AshwinDeshpande96/Hierarchical-Softmax/blob/master/hsm.ipynb)
