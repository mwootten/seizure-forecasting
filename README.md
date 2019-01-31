# Forecasting seizures using artificial neural networks

Current work on seizure prediction focuses on classifying segments at any given
time into "seizure" or "not seizure". Given that solutions to this problem are
rapidly improving, we ask another question: how long until a seizure occurs?

## Input

More precisely, we currently frame the question as "given that there will be a
seizure soon, when will it be?" We use the [CHB-MIT
dataset](https://physionet.org/pn6/chbmit/) to ask that question. We split the
recording up into five-second segments, then discard all segments that either:
- Overlap with a seizure (as this would incentivize the classifier to learn
  ictal patterns over all else)
- Has an indeterminate time until the next seizure (i.e. occurs after the last
  recorded seizure in a given file)

Note that the second requirement leads to substantial right-censorship. In the
future, we'll explore techniques from survival analysis to mitigate this issue.


## Feature selection

We're currently following [Tsiouris et al.
(2018)](https://doi.org/10.1016/j.compbiomed.2018.05.019) for our features.
Refer to that paper for details on what each of the features mean. The order of
the features in their vectors is documented in
`references/feature-descriptions.txt`

## Models

This repository currently contains a variety of models that attempt to predict
based solely on the features in a single model:
* Dummy (predicts the mean, only used as a baseline)
* Linear (ordinary least squares, no transformations)
* Multi-layer perceptron (with hidden layer sizes 60 and 10)
* Random forest (with 100 trees)
* Gradient boosted trees (all XGBoost defaults)

Due to the poor performance of all of these (see below), we intend to move on to
more powerful methods. Our next step will be using a convolutional network to
examine multiple adjacent five-second segments. We may also use an LSTM if the
convolutional network does not perform well.

## Evaluation

| Model         | R^2    | Adjusted R^2 |
| ------------- | ------ | ------------ |
| Dummy         | -0.132 | -0.209       |
| Linear        | -3.359 | -3.659       |
| MLP           | -5.765 | -6.229       |
| Random forest | -1.481 | -1.652       |
| XGBoost       | -1.647 | -1.829       |

See the `reports/figures` directory for predicted-actual plots.

## Running the code

* Install NumPy, then the requirements from the `requirements.txt` file. This is because pyEDFlib will refuse to install unless NumPy is already present.
* Run `create-directories.sh`
* Download the data from the CHB-MIT dataset with the `fetch-data.sh` script (requires `wget`)
* Calculate the feature values. This can be done with the `build_features.py` script in `src/features.py` for raw features, then `normalize_and_split.py` for scaling them for use in a neural network or other non-scale invariant model
* Run the individual models in the `models` directory. They are all split up into training and predicting steps.

## Authors

This code has all been written by Jeremy Angel and Matthew Wootten, except where
comments explicitly indicate otherwise.
