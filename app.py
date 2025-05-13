from flask import Flask, request, render_template
import pandas as pd
import ast
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load dataset
data1 = pd.read_csv(r"C:\Users\bhanu\OneDrive\Desktop\Projects\credits.csv")
data2 = pd.read_csv(r"C:\Users\bhanu\OneDrive\Desktop\Projects\movies.csv")
dataset = pd.merge(data1, data2, on='title')
dataset_new = dataset[['movie_id', 'title', 'crew', 'genres', 'keywords', 'production_companies', 'overview']].copy()
df = dataset_new.dropna().copy()

# Data Processing
def convert(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj)]
    except (ValueError, SyntaxError):
        return []

df['genres'] = df['genres'].apply(convert)
df['keywords'] = df['keywords'].apply(convert)
df['production_companies'] = df['production_companies'].apply(convert)

def convert_crew(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj) if i['job'] in ['Director', 'Writer']]
    except (ValueError, SyntaxError):
        return []

df['crew'] = df['crew'].apply(convert_crew)

def clean_text(obj):
    return " ".join([i.replace(" ", "") for i in obj])

df['crew'] = df['crew'].apply(clean_text)
df['keywords'] = df['keywords'].apply(clean_text)
df['production_companies'] = df['production_companies'].apply(clean_text)
df['genres'] = df['genres'].apply(lambda x: " ".join(x))
df['all'] = df['genres'] + " " + df['keywords'] + " " + df['overview'] + " " + df['crew']

df_new = df[['movie_id', 'title', 'all']].copy()
df_new['all'] = df_new['all'].str.lower()

stopword_set = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    words = word_tokenize(text)
    words = [word for word in words if word not in stopword_set]
    words = [ps.stem(word) for word in words]
    return " ".join(words)

df_new['all'] = df_new['all'].apply(preprocess_text)

cv = CountVectorizer(max_features=5000)
vectors_cv = cv.fit_transform(df_new['all'])
vectors_cv = vectors_cv.toarray()
cs_cv = cosine_similarity(vectors_cv)

def recommendation(movie):
    if movie not in df_new['title'].values:
        return ["Movie not found"]
    index = df_new[df_new['title'] == movie].index[0]
    dist = sorted(list(enumerate(cs_cv[index])), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = [df_new.iloc[i[0]]['title'] for i in dist]
    return recommended_movies

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        movie_name = request.form.get("movie")
        recommendations = recommendation(movie_name)
        return render_template("index.html", movie=movie_name, recommendations=recommendations)
    return render_template("index.html", movie=None, recommendations=[])

if __name__ == "__main__":
    app.run(debug=True)
