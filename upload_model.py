from huggingface_hub import create_repo, upload_folder

repo_id = "siddharth-1201/healthcare-chatbot-model"

# Repo create karega (agar already hai to skip)
create_repo(repo_id, exist_ok=True)

# Folder upload karega
upload_folder(
    folder_path="model/bert_model",
    repo_id=repo_id,
    repo_type="model"
)

print("Model uploaded successfully 🚀")