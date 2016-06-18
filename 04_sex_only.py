#!/usr/bin/env python
# -*- coding: utf-8 -*-

from jubakit.loader.csv import *
from jubakit.classifier import *

# 性別だけを使って推定してみます。

schema = Schema({
  'Survived': Schema.LABEL,
  'Sex': Schema.STRING,
}, Schema.IGNORE)

train_ds = Dataset(CSVLoader('train.csv'), schema)
test_ds = Dataset(CSVLoader('test.csv'), schema)

classifier = Classifier.run(Config(
  converter = {
    'string_rules': [{'key': 'Sex', 'type': 'str', 'sample_weight': 'bin', 'global_weight': 'bin'}]
  }
))

for _ in classifier.train(train_ds): pass

print('PassengerId,Survived')
for (idx, label, result) in classifier.classify(test_ds):
  print("{0},{1}".format(test_ds.get(idx)['PassengerId'], result[0][0]))
