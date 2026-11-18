import re
from dataclasses import dataclass
from pathlib import Path

# <h1 align="center"> Chapter title </h1>
H1_RE = re.compile(r"<h1[^>]*>\s*(.*?)\s*</h1>", re.DOTALL)

# Signature of a spirit message, written on two lines:
# **Erasto.** \
# **Paris, 1862.**
SIGNATURE_RE = re.compile(
    r"^\*\*(?P<who>[^*]+?)\*\*[ \t]*\\[ \t]*\n\*\*(?P<where>[^*]+?)\*\*[ \t]*$",
    re.MULTILINE,
)


@dataclass
class Section:
    chapter: str
    part: str  # e.g. "Instruções dos Espíritos" (empty when the file has no parts)
    title: str
    author: str  # short name taken from the signature, empty if not signed
    signature: str  # full signature line, e.g. "Erasto, discípulo de São Paulo"
    place_year: str  # e.g. "Paris, 1862"
    paragraphs: list[str]


def parse_document(path: Path) -> list[Section]:
    """Split a markdown file into sections, keeping title, part and author."""
    text = path.read_text(encoding="utf-8")

    # 1. The chapter title comes from the <h1> block
    chapter = ""
    h1_match = H1_RE.search(text)
    if h1_match:
        chapter = " ".join(h1_match.group(1).split())
        text = H1_RE.sub("", text, count=1)

    sections = []
    part = ""

    # 2. Sections are separated by lines containing only ---
    for block in re.split(r"^---\s*$", text, flags=re.MULTILINE):
        title = ""
        body_lines = []

        for line in block.strip().splitlines():
            if line.startswith("## "):
                title = line[3:].strip().rstrip(".")
            elif line.startswith("# "):
                part = line[2:].strip().rstrip(".")  # applies to the next sections too
            else:
                body_lines.append(line)

        body = "\n".join(body_lines).strip()
        if not body:
            continue

        # 3. Extract the signature (if any) and remove it from the body
        author = signature = place_year = ""
        sig_match = SIGNATURE_RE.search(body)
        if sig_match:
            signature = sig_match.group("who").strip().rstrip(".")
            place_year = sig_match.group("where").strip().rstrip(".")
            author = signature.split(",")[0].strip()
            body = (body[: sig_match.start()] + body[sig_match.end():]).strip()

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]

        sections.append(
            Section(
                chapter=chapter,
                part=part,
                title=title or chapter,  # sections without a heading use the chapter title
                author=author,
                signature=signature,
                place_year=place_year,
                paragraphs=paragraphs,
            )
        )

    return sections