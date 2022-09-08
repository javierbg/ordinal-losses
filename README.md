# Ordinal Losses
PyTorch implementation of ordinal losses for neural networks from the following papers:

* Classical losses
    * **CrossEntropy**, **MAE**, **MSE**
* Losses that promote ordinality
    * **OrdinalEncoding:** Cheng, Jianlin, Zheng Wang, and Gianluca Pollastri. "A neural network approach to ordinal regression." 2008 IEEE international joint conference on neural networks (IEEE world congress on computational intelligence). IEEE, 2008.
    * **WeightedKappa** (by default, Quadratic Weighted Kappa): de La Torre, Jordi, Domenec Puig, and Aida Valls. "Weighted kappa loss function for multi-class classification of ordinal data in deep learning." Pattern Recognition Letters 105 (2018): 144-154.
    * **CDW_CE:** Polat, Gorkem, et al. "Class Distance Weighted Cross-Entropy Loss for Ulcerative Colitis Severity Estimation." arXiv preprint arXiv:2202.05167 (2022).
* Losses that promote unimodality (soft)
    * **CO, CO2, HO2:** Albuquerque, Tomé, Ricardo Cruz, and Jaime S. Cardoso. "Ordinal losses for classification of cervical cancer risk." PeerJ Computer Science 7 (2021): e457.
    * **QUL_CE, QUL_HO:** Albuquerque, Tomé, Ricardo Cruz, and Jaime S. Cardoso. "Quasi-Unimodal Distributions for Ordinal Classification." Mathematics 10.6 (2022): 980.
    * **WassersteinUnimodal_KLDIV, WassersteinUnimodal_EMD:** to be published
* Losses with activations that force unimodality (hard)
    * **BinomialUnimodal_CE, BinomialUnimodal_MSE:** Costa, Joaquim F. Pinto, Hugo Alonso, and Jaime S. Cardoso. "The unimodal model for the classification of ordinal data." Neural Networks 21.1 (2008): 78-91.
    * **PoissonUnimodal:** Beckham, Christopher, and Christopher Pal. "Unimodal probability distributions for deep ordinal classification." International Conference on Machine Learning. PMLR, 2017.
    * **UnimodalNet:** to be published

## Install

```
pip3 install git+https://github.com/rpmcruz/ordinal-losses.git
```

## Pseudo-Usage

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
    # probabilities or classes
    probabilities, classes = loss.to_proba_and_classes(outputs)
```

* Your neural network should **not** perform any activation like softmax on the output.
* You should **not** perform any pre-processing on the labels, such as one-hot encoding.
* There is an example at `src/example.py`.
* Please notice that some losses have hyperparameters that you may fine-tune. Please read the code for more API information.
