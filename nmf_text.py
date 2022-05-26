import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

documents = pd.read_csv('news-data.csv')
print("Documentos:")
print(documents.head(), "\n")

#V = np.genfromtxt('news-data.csv', delimiter=' ', dtype=object)
V = documents
print(V, V.shape)

# use tfidf by removing tokens that don't appear in at least 50 documents
vect = TfidfVectorizer(stop_words='english')

# Fit and transform. Ajusta los datos primero y
# luego estandariza (centrado y escalado)
V = vect.fit_transform(documents.headline_text)
print(V, V.shape)


# Create an NMF instance: model
# the 10 components will be the topics
# random_state para reproducibilidad
k=10
model = NMF(n_components=k, random_state=0)

# Fit the model to TF-IDF.
model.fit(V)

# Get matrix W. Transform realiza la estandarización de centrado y escalado
# con la media y varianza hallados en fit_transform.
W = model.transform(V)

# Get matrix H
H = model.components_

print(V[0:1])
print(V[0:10])
print("V dimensiones:", V.shape, "\n")
print(W[0:3])
print("W dimensiones:", W.shape, "\n")
print(H[0:3])
print("H dimensiones:", H.shape, "\n")

print( np.matmul( V , np.transpose(H) ) )

W = W * ( np.matmul( V , np.transpose(H) ) / np.matmul(W,np.matmul(H,np.transpose(H))) )
print(W)

# Create a DataFrame: components_df
components_df = pd.DataFrame(model.components_, columns=vect.get_feature_names())
print(components_df)
