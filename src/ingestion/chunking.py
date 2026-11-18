import math


def split_by_words(text: str, max_chars: int) -> list[str]:
    """Split a long text into balanced pieces, always at word boundaries."""
    # Balanced pieces avoid leaving a tiny leftover at the end
    number_of_pieces = math.ceil(len(text) / max_chars)
    target_size = len(text) / number_of_pieces

    pieces = []
    current = ""

    for word in text.split():
        current = f"{current} {word}" if current else word
        if len(current) >= target_size and len(pieces) < number_of_pieces - 1:
            pieces.append(current)
            current = ""

    if current:
        pieces.append(current)
    return pieces


def chunk_paragraphs(paragraphs: list[str], max_chars: int = 5000) -> list[str]:
    """Pack whole paragraphs into chunks of up to max_chars characters."""
    chunks = []
    current = ""

    for paragraph in paragraphs:
        # A single paragraph bigger than the limit is split by words
        if len(paragraph) > max_chars:
            pieces = split_by_words(paragraph, max_chars)
        else:
            pieces = [paragraph]

        for piece in pieces:
            if current and len(current) + 2 + len(piece) > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = f"{current}\n\n{piece}" if current else piece

    if current:
        chunks.append(current)
    return chunks