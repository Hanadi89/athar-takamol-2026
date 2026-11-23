# أثر تكامل — MiniLM optimized for Render (2 GB RAM)

## Render settings (IMPORTANT)

- **Build Command:** `pip install -r requirements.txt && python download_model.py`
- **Start Command:** `python -u app.py`
- **Instance:** 2 GB RAM

Upload `app.py`, `requirements.txt`, `download_model.py`, and the two logo files to the GitHub repository. Do **not** upload the `models/` directory; Render downloads it during the build.

The first deployment may take longer during **Build** as MiniLM downloads. The web app then loads the model from local files on the first matching request, without downloading it from Hugging Face. The first request still has model initialization time, but should no longer wait on a network download.

If Render's build cannot download the model, the build fails with a visible error rather than leaving visitors waiting. Logs show model load and matching times.
