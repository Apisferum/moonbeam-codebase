import os
from huggingface_hub import snapshot_download

# Resolve target directory dynamically in the workspace root
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
local_dir = os.path.join(workspace_root, "Moonbeam Pretrained Weights")

snapshot_download(
    repo_id="guozixunnicolas/moonbeam-midi-foundation-model",
    local_dir=local_dir
)