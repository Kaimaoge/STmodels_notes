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

<br>

References
--------------
Franceschi, Jean-Yves, Aymeric Dieuleveut, and Martin Jaggi. "Unsupervised scalable representation learning for multivariate time series." Neurips 2019.

Eldele, Emadeldeen, Mohamed Ragab, Zhenghua Chen, Min Wu, Chee Keong Kwoh, Xiaoli Li, and Cuntai Guan. "Time-series representation learning via temporal and contextual contrasting." IJCAI 2021.


