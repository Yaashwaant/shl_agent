"""
Build vector index from catalog JSON.
Run this after build_catalog.py.
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from app.services.vector_store import get_vector_store


def main():
    vs = get_vector_store()
    catalog = vs.load_catalog()
    if not catalog:
        print("ERROR: No catalog data found. Run build_catalog.py first.")
        sys.exit(1)
    print(f"Building vector index for {len(catalog)} items...")
    vs.build_index(force=True)
    print("Vector index built successfully!")


if __name__ == "__main__":
    main()
