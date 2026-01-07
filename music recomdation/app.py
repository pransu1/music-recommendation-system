import os
from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_ROOT, 'models')

app = Flask(__name__, static_folder='static')

# Load models once at startup
vec = None
nn = None
kmeans = None
songs = None

def init_models():
    global vec, nn, kmeans, songs
    vec = joblib.load(os.path.join(MODEL_DIR, 'vectorizer.joblib'))
    nn = joblib.load(os.path.join(MODEL_DIR, 'nn.joblib'))
    kmeans = joblib.load(os.path.join(MODEL_DIR, 'kmeans.joblib'))
    # load dataframe saved by train.py
    songs = joblib.load(os.path.join(MODEL_DIR, 'songs_df.joblib'))


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json() or {}
    model = data.get('model', 'content')
    query = data.get('query', '')
    n = int(data.get('n', 10))

    if not query:
        return jsonify({'error': 'query is required'}), 400

    qv = vec.transform([query])
    # Ensure we return at least 30 recommendations unless the dataset is smaller
    min_required = 30
    n_req = max(n, min_required)
    total_songs = len(songs)

    results = []
    if model == 'content':
        neighbors_to_request = min(n_req, total_songs)
        distances, indices = nn.kneighbors(qv, n_neighbors=neighbors_to_request)
        indices = indices[0]
        for idx in indices[:min(n_req, len(indices))]:
            row = songs.iloc[idx]
            results.append({'Name': row['Name'], 'Artist': row['Artist'], 'Album': row['Album'], 'Popularity': int(row.get('Popularity', 0))})

        # If still fewer than requested (rare), pad with top popular songs
        if len(results) < n_req:
            needed = n_req - len(results)
            popular = songs.sort_values(by='Popularity', ascending=False)
            for _, row in popular.iterrows():
                entry = {'Name': row['Name'], 'Artist': row['Artist'], 'Album': row['Album'], 'Popularity': int(row.get('Popularity', 0))}
                if entry not in results:
                    results.append(entry)
                if len(results) >= n_req:
                    break

    elif model == 'cluster':
        cluster = int(kmeans.predict(qv)[0])
        if 'cluster' in songs.columns:
            cluster_songs = songs[songs['cluster'] == cluster]
        else:
            # fallback to on-the-fly compute (slower)
            cluster_songs = songs[songs.apply(lambda r: int(kmeans.predict(vec.transform([r['text']]))[0]) == cluster, axis=1)]

        cluster_songs = cluster_songs.sort_values(by='Popularity', ascending=False)
        for _, row in cluster_songs.head(n_req).iterrows():
            results.append({'Name': row['Name'], 'Artist': row['Artist'], 'Album': row['Album'], 'Popularity': int(row.get('Popularity', 0))})

        # If cluster doesn't have enough songs, pad with most popular songs overall
        if len(results) < n_req:
            needed = n_req - len(results)
            popular = songs.sort_values(by='Popularity', ascending=False)
            # Avoid duplicates by Name/Artist
            existing = set((r['Name'], r['Artist']) for r in results)
            for _, row in popular.iterrows():
                key = (row['Name'], row['Artist'])
                if key in existing:
                    continue
                results.append({'Name': row['Name'], 'Artist': row['Artist'], 'Album': row['Album'], 'Popularity': int(row.get('Popularity', 0))})
                existing.add(key)
                if len(results) >= n_req:
                    break
    else:
        return jsonify({'error': 'unknown model'}), 400

    return jsonify({'results': results})


if __name__ == '__main__':
    init_models()
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
