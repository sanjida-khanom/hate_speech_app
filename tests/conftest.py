import os
import sys

# Tests never need the 418 MB weights: always use the stub backend.
os.environ["HSD_BACKEND"] = "stub"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
