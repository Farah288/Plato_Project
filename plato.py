"""
Starting point for a Plato federated learning training session.
"""

import os
import sys # ADD THIS IMPORT

# --- ADD THESE THREE LINES ---
# Add the project's root directory to the Python path 
# so custom modules (like 'DR') can be imported.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
# -----------------------------

from plato.servers import registry as server_registry

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    server = server_registry.get()
    server.run()


if __name__ == "__main__":
    main()
