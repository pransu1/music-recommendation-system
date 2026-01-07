# Music Recommendation Demo

This project trains two simple recommender models from `songs.csv` and exposes a small web UI.

Steps

1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Train models (this reads `songs.csv` in the project root):

```bash
python train.py
```

3. Run the web server and open the UI:

```bash
python app.py
# then open http://127.0.0.1:5000
```

Usage

- Enter sample lyrics, song title, artist name or keywords and choose a model.
- `content` uses TF-IDF + nearest neighbors to find similar songs.
- `cluster` groups songs by text similarity and returns top songs from the predicted cluster.

Notes

- The pipeline is intentionally simple and intended as a starting point. You can swap in neural embeddings, add a genre column, or use collaborative signals for more personalized recommendations.
