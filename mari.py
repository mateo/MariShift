from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import json

# Import texts to JSON datastore
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


# Flatten Text
flat = {"Babylon": {}, "Mari": {}}
for city in ["Babylon", "Mari"]:
	flat[city].update({"city": [word for text in texts[city] for line in text["text"] for word in line]})
	flat[city].update({"char": [c for text in texts[city] for line in text["text"] for word in line for c in word if c in "aàábdeèéghiíìjklmnpqrsṣštṭuùúwyz"]})

# Normalize Character Based Datasets
if len(flat["Babylon"]["char"]) > len(flat["Mari"]["char"]):
	ncB = flat["Babylon"]["char"][0:len(flat["Mari"]["char"])]
	ncM = flat["Mari"]["char"]
if len(flat["Babylon"]["char"]) < len(flat["Mari"]["char"]):
	ncM = flat["Mari"]["char"][0:len(flat["Babylon"]["char"])]
	ncB = flat["Babylon"]["char"]

# Normalize Word Based Datasets
if len(flat["Babylon"]["city"]) > len(flat["Mari"]["city"]):
	nwB = flat["Babylon"]["city"][:len(flat["Mari"]["city"])]
	nwM = flat["Mari"]["city"]
if len(flat["Babylon"]["city"]) < len(flat["Mari"]["city"]):
	nwM = flat["Mari"]["city"][:len(flat["Babylon"]["city"])]
	nwB = flat["Babylon"]["city"]

sv.fit(ncM + ncB)

# Build the Models
dataset = {"Babylon": {}, "Mari": {}}
dataset["Babylon"].update({"Count Vector": cv.transform(nwB)})
dataset["Babylon"].update({"Tfidf Vector": tv.transform(nwB)})
dataset["Babylon"].update({"Shift Vector": sv.transform(ncB)})
dataset["Mari"].update({"Count Vector": cv.transform(nwM)})
dataset["Mari"].update({"Tfidf Vector": tv.transform(nwM)})
dataset["Mari"].update({"Shift Vector": sv.transform(ncM)})

# Train the Neural Net on the Vectorized texts
X_train, X_test, y_train, y_test = train_test_split(dataset["Babylon"]["Shift Vector"], test_size=0.25)
print(y_test.shape)
# Initialize the Neural Net
gnb = GaussianNB()

# Test the Neural Net Against the Dataset
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
	% (X_test.shape[0], (y_test != y_pred).sum()))
