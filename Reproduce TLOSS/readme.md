<br>

About unsupervised time series learning
--------------

Unsupervised learning is an approach in which a model **learns patterns, relationships, or structures** in data without the need for explicit labels or target outputs. In unsupervised learning time series learning, the studies primarily focus on using time series as input and employing self-learning techniques to obtain representations that can be applied to various downstream tasks such as forecasting, classification and anomaly detection. This approach **aims to accelerate the learning speed and performance of downstream tasks while endowing unsupervised models with strong transferability**. 

<br>

About TLOSS
--------------

TLOSS is a landmark model in unsupervised time series representation learning, which is proposed in [Neurips 19](https://proceedings.neurips.cc/paper/2019/hash/53c6de78244e9f528eb3e1cda69699bb-Abstract.html). Drawing inspiration from Word2Vec, it formulates unsupervised learning tasks for deep learning models. In this project, I will provide a brief introduction to the model and re-implement it based on [a relevent study](https://github.com/emadeldeen24/TS-TCC). Furthermore, I will conduct testing on datasets that were not evaluated in the original study.

<br>

How to understand unsupervised time series model?
--------------

In unsupervised learning, we do not have output labels to provide learning signals that guide the model on how to learn. So, how do we train a model to generate the desired representations? We must make certain assumptions, and these assumptions are based on some general properties of time series. We then design training methods on these assumptions to train the model. Therefore, it is essential to first examine the underlying assumptions behind a methodology and understand how it implements and validates these assumptions. I have found that this approach is a promising entry point for understanding unsupervised learning models. 

<br>

Assumption behind TLOSS
--------------

**The representation of a time series within a time window should be similar to that of any of its subseries. Conversely, its representation should be dissimilar to that of another randomly selected time series.**

<p align="center">
<img align="middle" src="https://github.com/Kaimaoge/STmodels_notes/raw/main/Reproduce%20TLOSS/image/tloss.png" width="500" />
</p>

The figure above illustrates how Tloss selects the target time series, represented by $x_{ref}$ and marked in green, along with the positive time series $x_{pos}$ marked in cyan, and the negative time series $x_{neg}$. marked in red. The motivation behind this configuration draws inspiration from the field of Natural Language Processing (NLP). It is based on the notion that the representation of a word's context should exhibit proximity to the word itself, yet distance from randomly chosen words. In this analogy, $x_{pos}$ can be considered as a word, and $x_{ref}$ as the corresponding context (a sentence, perhaps), and $x_{neg}$ as another word originating from a different context.

I believe this assumption is not entirely reasonable as different time series can have very similar contexts. For example, in spatiotemporal data, the time series of certain adjacent spatial points can exhibit similarities in their contexts. However, Tloss still makes a significant contribution by introducing the concept of positive and negative samples, akin to contrastive learning, to address the problem of unsupervised learning in time series analysis.

<br>

Learn the assumption
--------------

To enable a system to learn the prior knowledge embedded in the assumption, it is necessary to design a loss function and determine the sampling method for the input samples of the loss function. The loss function for Tloss is defined as follows:

$$ - log \left( \sigma \left(f(x_{ref}; \theta)^T f(x_{pos}; \theta)  \right) \right) - \sum^K_{k=1} log \left( \sigma \left(-f(x_{ref}; \theta)^T f(x_{neg, k}; \theta)\right) \right)  $$


<br>

References
--------------
Franceschi, Jean-Yves, Aymeric Dieuleveut, and Martin Jaggi. "Unsupervised scalable representation learning for multivariate time series." Neurips 2019.

Eldele, Emadeldeen, Mohamed Ragab, Zhenghua Chen, Min Wu, Chee Keong Kwoh, Xiaoli Li, and Cuntai Guan. "Time-series representation learning via temporal and contextual contrasting." IJCAI 2021.



