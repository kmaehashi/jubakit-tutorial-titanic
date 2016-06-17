#!/usr/bin/env python
# -*- coding: utf-8 -*-

from jubakit.loader.csv import *
from jubakit.classifier import *

schema = Schema({
  'Survived': Schema.LABEL,
}, Schema.INFER)

train_ds = Dataset(CSVLoader('train.csv'), schema)
test_ds = Dataset(CSVLoader('test.csv'), schema)

classifier = Classifier.run(Config())

for _ in classifier.train(train_ds): pass

print('PassengerId,Survived')
for (idx, label, result) in classifier.classify(test_ds):
  print("{0},{1}".format(test_ds.get(idx)['PassengerId'], result[0][0]))
