import argparse

import chromadb

from config import COLLECTION_NAME, DB_DIR

parser = argparse.ArgumentParser(description="List the sections stored in the database")
parser.add_argument("--author", help="only sections signed by this author (e.g. Erasto)")
args = parser.parse_args()

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection(COLLECTION_NAME)

where = {"author": args.author} if args.author else None
metadatas = collection.get(where=where, include=["metadatas"])["metadatas"]

# Many chunks belong to the same section, so keep each section only once
sections = {}
for meta in metadatas:
    key = (meta["source"], meta["section_index"])
    sections[key] = meta

for (_, _), meta in sorted(sections.items()):
    line = f"{meta['book']} | {meta['chapter']} | {meta['section']}"
    if meta["author"]:
        line += f" | {meta['signature']}, {meta['place_year']}"
    print(line)

print(f"\n{len(sections)} section(s)")