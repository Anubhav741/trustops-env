import json
import os
from env import TrustOpsEnv

def load_dataset(filepath=None):
    if filepath is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(base_dir, "data", "dataset.json")
    with open(filepath, "r") as f:
        return json.load(f)

def get_easy_detection_task() -> TrustOpsEnv:
    data = load_dataset()
    subset = [d for d in data if d.get("task") == "easy_detection"]
    return TrustOpsEnv(dataset=subset)

def get_medium_classification_task() -> TrustOpsEnv:
    data = load_dataset()
    subset = [d for d in data if d.get("task") == "medium_classification"]
    return TrustOpsEnv(dataset=subset)

def get_hard_contextual_task() -> TrustOpsEnv:
    data = load_dataset()
    subset = [d for d in data if d.get("task") == "hard_contextual"]
    return TrustOpsEnv(dataset=subset)
