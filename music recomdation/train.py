import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
import joblib


def main(csv_path='songs.csv', out_dir='models'):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # ensure required columns exist
    for col in ['Name', 'Artist', 'Album', 'Popularity', 'Lyrics']:
        if col not in df.columns:
            raise RuntimeError(f"Missing required column: {col}")

    df = df.fillna('')
    # Combine text fields for content-based model
    df['text'] = (df['Name'].astype(str) + ' ' + df['Artist'].astype(str) + ' ' +
                  df['Album'].astype(str) + ' ' + df['Lyrics'].astype(str))

    print('Fitting TF-IDF vectorizer...')
    vectorizer = TfidfVectorizer(max_features=20000, stop_words='english')
    X = vectorizer.fit_transform(df['text'])

    print('Fitting NearestNeighbors (content-based)...')
    nn = NearestNeighbors(n_neighbors=10, metric='cosine', n_jobs=-1)
    nn.fit(X)

    n_songs = len(df)
    n_clusters = min(20, max(2, n_songs // 50))
    print(f'Fitting MiniBatchKMeans with {n_clusters} clusters...')
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    # MiniBatchKMeans accepts sparse input
    kmeans.fit(X)

    # assign cluster labels to dataframe for quick lookup
    print('Assigning cluster labels...')
    df['cluster'] = kmeans.predict(X)

    print('Saving models and data...')
    joblib.dump(vectorizer, os.path.join(out_dir, 'vectorizer.joblib'))
    joblib.dump(nn, os.path.join(out_dir, 'nn.joblib'))
    joblib.dump(kmeans, os.path.join(out_dir, 'kmeans.joblib'))
    # Save dataframe (with cluster labels) for quick lookup
    joblib.dump(df, os.path.join(out_dir, 'songs_df.joblib'))

    print('Done. Models saved to', out_dir)


if __name__ == '__main__':
    main()
