from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="superkart_project/deployment",     # the local folder containing your files
    repo_id="JyalaHarsha-2025/MLOPS_Superkart_Sales_Forcast",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
