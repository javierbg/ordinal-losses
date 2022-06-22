# Ordinal Losses
PyTorch implementation of ordinal losses for neural networks from the following papers:

* Beckham, Christopher, and Christopher Pal. "Unimodal probability distributions for deep ordinal classification." International Conference on Machine Learning. PMLR, 2017.
* Cheng, Jianlin, Zheng Wang, and Gianluca Pollastri. "A neural network approach to ordinal regression." 2008 IEEE international joint conference on neural networks (IEEE world congress on computational intelligence). IEEE, 2008.
* da Costa, Joaquim F. Pinto, Hugo Alonso, and Jaime S. Cardoso. "The unimodal model for the classification of ordinal data." Neural Networks 21.1 (2008): 78-91.
* Albuquerque, Tomé, Ricardo Cruz, and Jaime S. Cardoso. "Ordinal losses for classification of cervical cancer risk." PeerJ Computer Science 7 (2021): e457.

## Usage

Pseudo-code usage:

```python
import ordinal_losses
loss = ordinal_losses.OrdinalEncoder(K=10)
model = BuildModel(n_outputs=loss.how_many_outputs())
for X, Y in train:
    outputs = model(X)
    outputs = loss.process(outputs)
    loss_value = loss.compute_loss(outputs)
    probabilities = loss.to_proba(outputs)
    predicted_classes = loss.to_classes(probabilities)
```

Notice that:

* They receive the labels as scalars. No one-hot encoding should be performed. Any encoding will be performed by the loss, if necessary.
* Losses may require different number of output neurons from your network. Therefore, it is necessary to ask them how many outputs are required.
* The `process()` method must be called before calling `compute_loss()` or `to_proba()`. This is due to efficiency reasons.

## Install

```
pip3 install git+https://github.com/rpmcruz/ordinal-losses.git
```
