import yaml
from pathlib import Path

def load_eval_cases(file_path: str):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    cases = load_eval_cases("evals/cases.yaml")
    print(f"Loaded {len(cases)} evaluation cases:")
    for case in cases:
        print(case)