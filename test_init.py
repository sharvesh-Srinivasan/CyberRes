import sys
import os

# Ensure we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fastapi_app import init_pipeline

if __name__ == "__main__":
    print("Running init_pipeline() directly...")
    init_pipeline()
    print("Done!")
