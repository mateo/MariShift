from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import json
import numpy as np
import matplotlib.pyplot as plt

# Import Texts to JSON Datastore
filename = "texts.json"
if filename:
	with open(filename, 'r') as f:
		t = json.load(f)

# Organize the Corpora and collect the vocabularies
texts = {"Babylon": [], "Mari": [], "Diplomats": []}
vocab = {"Babylon": [], "Mari": []}
for text in t["texts"]:
	if text["origin"] == "Babylon" and text["author-origin"] == "Babylon":
		texts["Babylon"].append(text)
		vocab["Babylon"].append(text["text"])
	elif text["origin"] == "Mari" and text["author-origin"] == "Mari":
		texts["Mari"].append(text)
		vocab["Mari"].append(text["text"])
	else:
		texts["Diplomats"].append(text)

# Flatten the Vocabularies and Remove Restorations and Notation from Character Database
vocab.update({"flat": [word for city in ["Babylon", "Mari"] for tablet in vocab[city] for line in tablet for word in line]})
vocab.update({"char": [c for word in vocab["flat"] for c in word if c in "aàábdeèéghiíìjklmnpqrsṣštṭuùúwyz"]})

# Initialize the Transforms
cv = CountVectorizer()
tv = TfidfVectorizer()
sv = TfidfVectorizer(analyzer='char')

# Build the Vocabularies
cv.fit(vocab["flat"])
tv.fit(vocab["flat"])
sv.fit(vocab["char"])

# Flatten Text
flat = {"Babylon": {}, "Mari": {}, "Diplomats": {}}
for city in ["Babylon", "Mari", "Diplomats"]:
	flat[city].update({"city": [word for text in texts[city] for line in text["text"] for word in line]})
	flat[city].update({"char": [c for text in texts[city] for line in text["text"] for word in line for c in word if c in "aàábdeèéghiíìjklmnpqrsṣštṭuùúwyz"]})

# Normalize Character Based Datasets
if len(flat["Babylon"]["char"]) > len(flat["Mari"]["char"]):
	ncB = flat["Babylon"]["char"][0:len(flat["Mari"]["char"])]
	ncM = flat["Mari"]["char"]
if len(flat["Babylon"]["char"]) < len(flat["Mari"]["char"]):
	ncM = flat["Mari"]["char"][0:len(flat["Babylon"]["char"])]
	ncB = flat["Babylon"]["char"]
if len(flat["Babylon"]["char"]) > len(flat["Diplomats"]["char"]):
	ncB = flat["Babylon"]["char"][0:len(flat["Diplomats"]["char"])]
	ncM = flat["Mari"]["char"][0:len(flat["Diplomats"]["char"])]
	ncD = flat["Diplomats"]["char"]
if len(flat["Babylon"]["char"]) < len(flat["Diplomats"]["char"]):
	ncD = flat["Diplomats"]["char"][0:len(flat["Babylon"]["char"])]

# Normalize Word Based Datasets
if len(flat["Babylon"]["city"]) > len(flat["Mari"]["city"]):
	nwB = flat["Babylon"]["city"][:len(flat["Mari"]["city"])]
	nwM = flat["Mari"]["city"]
if len(flat["Babylon"]["city"]) < len(flat["Mari"]["city"]):
	nwM = flat["Mari"]["city"][:len(flat["Babylon"]["city"])]
	nwB = flat["Babylon"]["city"]
if len(flat["Babylon"]["city"]) > len(flat["Diplomats"]["city"]):
	nwB = flat["Babylon"]["city"][0:len(flat["Diplomats"]["city"])]
	nwM = flat["Mari"]["city"][0:len(flat["Diplomats"]["city"])]
	nwD = flat["Diplomats"]["city"]
if len(flat["Babylon"]["city"]) < len(flat["Diplomats"]["city"]):
	nwD = flat["Diplomats"]["city"][0:len(flat["Babylon"]["city"])]

# Build the Models
dataset = {"Babylon": {}, "Mari": {}, "Diplomats": {}}
dataset["Diplomats"].update({"Count Vector": cv.transform(nwD)})
dataset["Diplomats"].update({"Tfidf Vector": tv.transform(nwD)})
dataset["Diplomats"].update({"Shift Vector": sv.transform(ncD)})
dataset["Babylon"].update({"Count Vector": cv.transform(nwB)})
dataset["Babylon"].update({"Tfidf Vector": tv.transform(nwB)})
dataset["Babylon"].update({"Shift Vector": sv.transform(ncB)})
dataset["Mari"].update({"Count Vector": cv.transform(nwM)})
dataset["Mari"].update({"Tfidf Vector": tv.transform(nwM)})
dataset["Mari"].update({"Shift Vector": sv.transform(ncM)})

# Collapse Dataset to 1D
pca = PCA(n_components=10)
pca.fit(dataset["Babylon"]["Tfidf Vector"].todense())
pcaB = pca.transform(dataset["Babylon"]["Tfidf Vector"].todense())
pca.fit(dataset["Mari"]["Tfidf Vector"].todense())
pcaM = pca.transform(dataset["Mari"]["Tfidf Vector"].todense())
pca.fit(dataset["Diplomats"]["Tfidf Vector"].todense())
pcaD = pca.transform(dataset["Diplomats"]["Tfidf Vector"].todense())

# Define dataset to split
X = np.concatenate((pcaM, pcaD))
y = np.concatenate((np.ones(pcaM.shape[0]) - 1, np.ones(pcaM.shape[0])))

# Split the Datasets for Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Initialize the Neural Net
gnb = GaussianNB()
neigh = KNeighborsClassifier(n_neighbors=3)

# Test the Neural Net Against the Dataset
#y_pred = gnb.fit(X_train, y_train).predict(X_test)
y_pred = neigh.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
	% (X_test.shape[0], (y_test != y_pred).sum()))

# Plot PCA Decomposed Vectors
plt.scatter(pcaM[:,1], pcaM[:,2], c = "blue")
plt.scatter(pcaD[:,1], pcaD[:,2], c = "green")
#plt.scatter(pcaB[:,1], pcaB[:,2], c = "red")
#plt.scatter(pcaB[:,0], pcaB[:,1], c = "red")
#plt.scatter(pcaM[:,0], pcaM[:,1], c = "red")
#plt.scatter(pcaD[:,0], pcaD[:,1], c = "red")

plt.show()
