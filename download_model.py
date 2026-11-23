"""Download and save MiniLM during Render build, not during user requests."""
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEST = Path(__file__).resolve().parent / "models" / "minilm"
if not (DEST / "modules.json").is_file():
    print("Downloading MiniLM during build...", flush=True)
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(DEST))
print("MiniLM model files ready at", DEST, flush=True)
