from id3 import Id3Estimator, export_text, export_graphviz
import numpy as np
import pandas as pd

df = pd.read_csv('dataset.csv')

print('+++ CSV Data +++')
print(df.head())

df['outlook'] = df.outlook.map({'sunny': 0, 'overcast': 1, 'rain': 2})
df['temperature'] = df.temperature.map({'hot': 0, 'mild': 1, 'cool': 2})
df['humidity'] = df.humidity.map({'high': 0, 'normal': 1})
df['windy'] = df.windy.map({'weak': 0, 'strong': 1})

print('\n+++ CSV Data Change +++')
print(df.head())

print('\n+++ Data shape +++')
print(df.shape)

data = df.values
print('\n+++ Data values +++')
print(data)

data_train = data[:, :-1]
print('\n+++ Data train +++')
print(data_train)

data_label = data[:, -1:].flatten()
print('\n+++ Data label +++')
print(data_label)

clf = Id3Estimator()
clf.fit(data_train, data_label, check_input=True)

feature_names = [
    "outlook", "temperature", "humidity", "windy"
]

exported_text = export_text(clf.tree_, feature_names)

print(exported_text)

export_graphviz(clf.tree_, 'out.dot', feature_names)
