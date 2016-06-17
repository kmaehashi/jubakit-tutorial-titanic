#!/usr/bin/env python
# -*- coding: utf-8 -*-

from jubakit.loader.csv import *
from jubakit.classifier import *

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import classification_report

schema = Schema({
  'PassengerId': Schema.IGNORE,
  'Survived': Schema.LABEL,
  'Pclass': Schema.NUMBER,
  'Name': Schema.STRING,
  'Sex': Schema.STRING,
  'Age': Schema.NUMBER,
  'SibSp': Schema.NUMBER,
  'Parch': Schema.NUMBER,
  'Ticket': Schema.STRING,
  'Fare': Schema.NUMBER,
  'Cabin': Schema.STRING,
  'Embarked': Schema.STRING,
})

dataset = Dataset(CSVLoader('train.csv'), schema)

classifier = Classifier.run(Config())

y_true = []
y_pred = []
for train_idx, test_idx in StratifiedKFold(list(dataset.get_labels()), n_folds=10):
  classifier.clear()

  (train_ds, test_ds) = (dataset[train_idx], dataset[test_idx])
  for _ in classifier.train(train_ds): pass
  for (idx, label, result) in classifier.classify(test_ds):
    y_true.append(label)
    y_pred.append(result[0][0])

print(classification_report(y_true, y_pred))
