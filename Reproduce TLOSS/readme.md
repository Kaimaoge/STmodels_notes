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

where $\sigma$ is the sigmoid function. This loss pushes the computed representations to distinguish between $x_{\text{ref}}$ and $x_{\text{neg}}$, and to assimilate $x_{\text{ref}}$ and $x_{\text{pos}}$. $x_{\text{ref}}$, $x_{\text{pos}}$ and $x_{\text{neg}}$ are sampled randomly from the training dataset.

The learning structure of Tloss is not particularly special. It consists of a causal convolutional network, but it needs to map time series of different lengths to the same length for computing the loss function. The output of this causal network is then **given to a global max pooling layer squeezing the temporal dimension and aggregating all temporal information in a fixed-size vector**.

<br>

My implentation
--------------

Following Tloss, Eldele et al. proposed an unsupervised learning model named TS-TCC in IJCAI 2021. However, they did not evaluate their model on the same datasets. I am curious about the performance of Tloss on the datasets evaluated in the TS-TCC paper. Therefore, I reproduced a Tloss model based on the open-source code provided by the authors in their [TS-TCC repository](https://github.com/emadeldeen24/TS-TCC).

### Datasets

Fortunately, the authors of TS-TCC have provided their preprocessed datasets, which include sleep-EDF, HAR, and Epilepsy. You can now access the preprocessed datasets on the [Dataverse](https://researchdata.ntu.edu.sg/dataverse/tstcc/) associated with TS-TCC.

### Minor changes to the original paper

In the original paper, once Tloss is trained to obtain the representations, the parameters remain unchanged. The generated features are then used as inputs to downstream SVM classifiers. On the other hand, TS-TCC employs a pre-training then fine-tuning strategy, where the model's parameters are further fine-tuned on downstream tasks after pretraining. In this project, we adopt the finetuning approach of TS-TCC to train Tloss. We do not use an SVM classifier, but instead directly employ an MLP classifier at the top of the Tloss model. See models.causal_cnn.py

```python
class CausalCNNEncoder(torch.nn.Module):
    def __init__(self, configs):
        super(CausalCNNEncoder, self).__init__()
        causal_cnn = CausalCNN(
            configs.in_channels, configs.channels, configs.depth, configs.reduced_size, configs.kernel_size
        )
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
        linear = torch.nn.Linear(configs.reduced_size, configs.out_channels)
        self.network = torch.nn.Sequential(
            causal_cnn, reduce_size, squeeze, linear
        )
        self.logits = torch.nn.Linear(configs.out_channels, configs.num_classes)

    def forward(self, x):
        x = self.network(x)
        logits = self.logits(x)
        return x, logits
```

In the original Tloss open-source code, negative sample sampling was performed across all training samples. Due to the large sample size, the probability of sampling $x_{ref}$ itself is relatively small. Therefore, the authors did not impose restrictions to avoid sampling the time series containing $x_{ref}$. In this project, negative sample sampling is performed within each training batch. We have imposed certain constraints on the sampling process to avoid sampling the time series containing $x_{ref}$. See dataloader.augmentations.py

```python
samples = np.concatenate(
        [np.random.choice(
            np.delete(np.arange(batch_size), j),
            size=(configs.nb_random_samples, 1)
        ) for j in range(batch_size)], axis = 1
    )                 # It can perfectly avoid self selection
```

<br>

Results
--------------

| Notebook  | HAR       |       | Sleep-EDF   |     | Epilepsy     |    |
|-----------|--------|----------|--------|----------|--------|----------|
| Baseline  | ACC    | MF1     | ACC    | MF1     | ACC    | MF1     |
| TS-TCC    | 90.37±0.34 | 90.38±0.39 | 83.00±0.71 | 73.57±0.74 | 97.23±0.10 | 95.54±0.08 |
| Tloss     | 87.89 | 88.03 | 84.05 | 75.60 | 97.65 | 95.66 |

From the results, it can be seen that Tloss is not worse than TS-TCC. However, more experiments are needed for a detailed comparison.

<br>

To be continued
--------------

  - Implementation of other unsupervised time series learning models
  - Experiments on settings more suitable for unsupervised learning (limited label, transfer learning, etc)
  - .......


<br>

References
--------------
Franceschi, Jean-Yves, Aymeric Dieuleveut, and Martin Jaggi. "Unsupervised scalable representation learning for multivariate time series." Neurips 2019.

Eldele, Emadeldeen, Mohamed Ragab, Zhenghua Chen, Min Wu, Chee Keong Kwoh, Xiaoli Li, and Cuntai Guan. "Time-series representation learning via temporal and contextual contrasting." IJCAI 2021.



