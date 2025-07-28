# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# model.save("./local_model")
# download_model.py

from sentence_transformers import SentenceTransformer

# Model name
model_name = 'all-MiniLM-L6-v2'

# Load from Hugging Face and save locally
print(f"Downloading model: {model_name} ...")
model = SentenceTransformer(model_name)

save_path = "local_model"
model.save(save_path)

print(f"✅ Model downloaded and saved to: {save_path}")
