# Ordinal Losses
PyTorch implementation of ordinal losses for neural networks from the following papers:

* Beckham, Christopher, and Christopher Pal. "Unimodal probability distributions for deep ordinal classification." International Conference on Machine Learning. PMLR, 2017.
* Cheng, Jianlin, Zheng Wang, and Gianluca Pollastri. "A neural network approach to ordinal regression." 2008 IEEE international joint conference on neural networks (IEEE world congress on computational intelligence). IEEE, 2008.
* da Costa, Joaquim F. Pinto, Hugo Alonso, and Jaime S. Cardoso. "The unimodal model for the classification of ordinal data." Neural Networks 21.1 (2008): 78-91.
* Polat, Gorkem, et al. "Class Distance Weighted Cross-Entropy Loss for Ulcerative Colitis Severity Estimation." arXiv preprint arXiv:2202.05167 (2022).
* Albuquerque, Tomé, Ricardo Cruz, and Jaime S. Cardoso. "Ordinal losses for classification of cervical cancer risk." PeerJ Computer Science 7 (2021): e457.
* Albuquerque, Tomé, Ricardo Cruz, and Jaime S. Cardoso. "Quasi-Unimodal Distributions for Ordinal Classification." Mathematics 10.6 (2022): 980.

If you use this package, please cite the last paper.

## Install

```
pip3 install git+https://github.com/rpmcruz/ordinal-losses.git
```

## Pseudo-code Usage

```python
import ordinal_losses
loss = ordinal_losses.OrdinalEncoder(K=10)
# different losses require different number of output neurons, therefore you
# should ask the loss for how many output neurons are necessary
model = BuildModel(n_outputs=loss.how_many_outputs())
for X, Y in train:
    outputs = model(X)
    loss_value = loss(outputs)
    # for evaluation purposes, use our methods to convert the outputs into
    # probabilities
    probabilities = loss.to_proba(outputs)
    predicted_classes = loss.to_classes(probabilities)
```

* Your neural network should **not** perform any activation like softmax on the output.
* You should **not** perform any pre-processing on the labels, such as one-hot encoding.
