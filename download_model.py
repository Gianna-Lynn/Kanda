from huggingface_hub import snapshot_download

print("Starting download of Qwen2.5-7B-Instruct model...")
print("This is about 15GB and may take some time.")

model_id = "Qwen/Qwen2.5-7B-Instruct"

# Download to the 'models' folder in the current directory
snapshot_download(
    repo_id=model_id, 
    local_dir="./models/Qwen2.5-7B-Instruct",
    local_dir_use_symlinks=False
)

print("Download complete!")