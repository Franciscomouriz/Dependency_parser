# Dependency Parser for English

## Getting Started

To get the model up and running, simply initiate the model selection process by calling:

```python
select_model(n_features, new_model, new_samples)
```

## Function Parameters

- `n_features`: Defines the number of features to be used from the stack and buffer. Take into account that the number of features is 4*n_features. By defect it is set to 2.
  - 2 words of the stack
  - 2 words of the buffer
  - 2 POS tags corresponding to the words of the stack
  - 2 POS tags corresponding to the words of the buffer

- `new_model`: Determines whether to train a new model regardless of an existing one.
  - `True` initiates training a new model and overwrites the existing one.
  - `False` uses the existing model if available.
  - The default setting is `False`.

- `new_samples`: Determines whether to create new samples of the given dataset.
  - `True` creates new samples and overwrites the existing ones.
  - `False` uses the existing samples if available.
  - The default setting is `False`.

## Return Value

This function returns a `DependencyParser` object, encapsulating the trained model and tokenizer. 

## Evaluate the model

The DependencyParser class has a method to evaluate the model. It uses the test data to evaluate the model and returns the accuracy and the loss of the model.

```python
dependencyParser.evaluate_model()
loss: 1.8781 - output1_loss: 0.7728 - output2_loss: 1.1053 - output1_accuracy: 0.8154 - output2_accuracy: 0.7991
```

This function only evaluate sample per sample. In order to evaluate if the dependency tree predicted is correct, you need to generate the predictions and compare them with the gold standard.

```python
predictions = dependencyParser.predict(dependencyParser.processData.test_data["dataframes"], n_features=2)
dependencyParser.conllu_evaluation(predictions, n_features=2)
```

This functions is always executed when the model is trained or loaded. It is not necessary to execute it again.

The results are stored in the folder `evaluation/{n_features}/`. In this folder you can find three files:
 - `original_trees.conllu`: contains the dependency trees of the test data.
 - `predicted_trees.conllu`: contains the dependency trees predicted by the model.
 - `results.txt`: contains the results of the evaluation.
  
The results file contains the following information:

```
LAS F1 Score: 53.51
MLAS Score: 43.27
BLEX Score: 45.12

Metric     | Precision |    Recall |  F1 Score | AligndAcc
-----------+-----------+-----------+-----------+-----------
Tokens     |    100.00 |    100.00 |    100.00 |
Sentences  |    100.00 |    100.00 |    100.00 |
Words      |    100.00 |    100.00 |    100.00 |
UPOS       |    100.00 |    100.00 |    100.00 |    100.00
XPOS       |    100.00 |    100.00 |    100.00 |    100.00
UFeats     |    100.00 |    100.00 |    100.00 |    100.00
AllTags    |    100.00 |    100.00 |    100.00 |    100.00
Lemmas     |    100.00 |    100.00 |    100.00 |    100.00
UAS        |     65.09 |     65.09 |     65.09 |     65.09
LAS        |     53.51 |     53.51 |     53.51 |     53.51
CLAS       |     51.19 |     40.33 |     45.12 |     40.33
MLAS       |     49.09 |     38.68 |     43.27 |     38.68
BLEX       |     51.19 |     40.33 |     45.12 |     40.33
```

## Model Training and Storage

If `new_model` is set to `False` or no model is available for the chosen number of features, the function will automatically train a new model. The model and tokenizer are then saved in the `models` directory.

## Directory Structure

The directory structure is as follows:

```
.
├── README.md
├── evaluation
│   ├── 1_features
│   │   ├── original_trees.conllu
│   │   ├── predicted_trees.conllu
│   │   └── results.txt
|   ├── 2_features
│   |   └── ...
|   └── conll18_ud_eval.py
├── models
│   ├── 1_features_parser.h5
│   ├── 2_features_tokenizer.h5
│   ├── ...
|   └── tokenizer.pickle
├── samples
│   ├── dev_samples.json
│   ├── test_samples.json
│   └── train_samples.json
├── data_dictionaries.py
├── dependency_parser.py
├── oracle.py
├── Dependency_Parser.ipynb
└── process_data.py
