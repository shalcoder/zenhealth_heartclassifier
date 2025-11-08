# scripts/save_model.py
import subprocess
subprocess.run(["python", "scripts/train_model.py"])
print("Model training script executed and model saved.")