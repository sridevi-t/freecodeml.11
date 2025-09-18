import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from zipfile import ZipFile
import urllib.request
import os

url = "https://cdn.freecodecamp.org/project-data/books/book-crossings.zip"
filename = "book-crossings.zip"

if not os.path.exists(filename):
    urllib.request.urlretrieve(url, filename)

with ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall("book_data")

books = pd.read_csv("book_data/Books.csv", encoding='latin-1', sep=';', on_bad_lines='skip')
ratings = pd.read_csv("book_data/Book-Ratings.csv", encoding='latin-1', sep=';', on_bad_lines='skip')

ratings = ratings[ratings['Book-Rating'] > 0]  # Only consider non-zero ratings
book_ratings = ratings.merge(books, on='ISBN')

pivot = book_ratings.pivot_table(index='User-ID', columns='Book-Title', values='Book-Rating').fillna(0)

book_features = pivot.T

model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(book_features.values)

def get_recommends(book_title: str):
    if book_title not in book_features.index:
        return [book_title, []]

    book_index = book_features.index.get_loc(book_title)
    distances, indices = model.kneighbors([book_features.iloc[book_index].values], n_neighbors=6)

    recommendations = []
    for i in range(1, len(distances[0])):  # skip the first, it's the book itself
        recommended_title = book_features.index[indices[0][i]]
        dist = round(distances[0][i], 3)
        recommendations.append([recommended_title, dist])

    return [book_title, recommendations]

recs = get_recommends("The Lovely Bones: A Novel")
print(recs)
