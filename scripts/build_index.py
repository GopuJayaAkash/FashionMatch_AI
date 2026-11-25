from fashionmatch.indexer import build_index

if __name__ == "__main__":
    count = build_index()
    print(f"Successfully indexed {count} products.")
