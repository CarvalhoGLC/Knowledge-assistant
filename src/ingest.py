import chromadb
import ollama

from config import BOOK_TITLES, COLLECTION_NAME, DATA_DIR, DB_DIR, EMBED_MODEL
from ingestion.chunking import chunk_paragraphs
from ingestion.parsing import parse_document


def build_context_header(book: str, chapter: str, title: str, author: str) -> str:
    """Text added before each chunk ONLY when generating its embedding."""
    header = f"{book} - {chapter}\n{title}"
    if author:
        header += f"\nAutor: {author}"
    return header


def main():
    client = chromadb.PersistentClient(path=DB_DIR)

    # Rebuild from scratch, so old chunks never stay in the database
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # the collection did not exist yet
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    files = sorted(DATA_DIR.glob("*.md"))
    if not files:
        print(f"No .md files found in {DATA_DIR}")
        return

    for file in files:
        book = BOOK_TITLES.get(file.stem, file.stem)
        sections = parse_document(file)

        ids, documents, texts_to_embed, metadatas = [], [], [], []

        for section_index, section in enumerate(sections):
            chunks = chunk_paragraphs(section.paragraphs)
            header = build_context_header(book, section.chapter, section.title, section.author)

            for chunk_index, chunk in enumerate(chunks):
                ids.append(f"{file.stem}-{section_index}-{chunk_index}")
                documents.append(chunk)  # clean text, shown in the results
                texts_to_embed.append(f"{header}\n\n{chunk}")  # text the vector is built from
                metadatas.append(
                    {
                        "source": file.name,
                        "book": book,
                        "chapter": section.chapter,
                        "part": section.part,
                        "section": section.title,
                        "author": section.author,
                        "signature": section.signature,
                        "place_year": section.place_year,
                        "section_index": section_index,
                        "chunk_index": chunk_index,
                    }
                )

        embeddings = ollama.embed(model=EMBED_MODEL, input=texts_to_embed)["embeddings"]
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"{file.name}: {len(sections)} sections, {len(ids)} chunks")

    print(f"\nTotal chunks in the database: {collection.count()}")


if __name__ == "__main__":
    main()