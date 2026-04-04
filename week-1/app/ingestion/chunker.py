from typing import Dict, List


def fixed_chunk(
    pages: List[Dict],
    chunk_size: int = 500,
    overlap: int = 100
) -> List[Dict]:
    """
    Fixed-size character chunking with overlap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[Dict] = []

    print("\n" + "=" * 70)
    print("FIXED CHUNKING")
    print(f"  chunk_size = {chunk_size} chars")
    print(f"  overlap    = {overlap} chars")
    print(f"  step       = {chunk_size - overlap} chars (chunk_size - overlap)")
    print("=" * 70)

    for page_data in pages:
        text = page_data["text"]
        source = page_data["source"]
        page = page_data["page"]

        print(f"\n  Page {page} | Total text length: {len(text)} chars")
        print(f"  {'~' * 60}")

        start = 0
        chunk_num = 1
        step = chunk_size - overlap

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": f"{source}_p{page}_c{chunk_num}_fixed",
                        "text": chunk_text,
                        "source": source,
                        "page": page,
                        "chunking_method": "fixed",
                    }
                )

                # Show overlap visually
                overlap_text = ""
                if chunk_num > 1 and overlap > 0:
                    overlap_text = f" | OVERLAP first {overlap} chars: \"{chunk_text[:overlap]}...\""

                print(f"\n    Chunk #{chunk_num}")
                print(f"      ID      : {source}_p{page}_c{chunk_num}_fixed")
                print(f"      Position: chars [{start} : {min(end, len(text))}]")
                print(f"      Length  : {len(chunk_text)} chars")
                if overlap_text:
                    print(f"      {overlap_text}")
                print(f"      Preview : \"{chunk_text[:100]}...\"" if len(chunk_text) > 100 else f"      Preview : \"{chunk_text}\"")

            start += step
            chunk_num += 1

    print(f"\n  TOTAL CHUNKS CREATED: {len(chunks)}")
    print("=" * 70)
    return chunks


def recursive_split_text(
    text: str,
    chunk_size: int,
    overlap: int
) -> List[str]:
    """
    Beginner-friendly recursive-ish strategy:
    1. Try grouping sentence-like parts.
    2. If one part is still too large, hard split it.
    """
    if len(text) <= chunk_size:
        return [text]

    parts = [part.strip() for part in text.split(". ") if part.strip()]
    chunks: List[str] = []
    current = ""

    print(f"    Sentences found: {len(parts)}")

    for i, part in enumerate(parts):
        if not part.endswith("."):
            part = part + "."

        candidate = f"{current} {part}".strip() if current else part

        if len(candidate) <= chunk_size:
            print(f"      Sentence {i+1}: \"{part[:60]}...\" -> MERGED (candidate: {len(candidate)} chars <= {chunk_size})" if len(part) > 60 else f"      Sentence {i+1}: \"{part}\" -> MERGED (candidate: {len(candidate)} chars <= {chunk_size})")
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
                print(f"      >>> CHUNK SAVED (length: {len(current.strip())} chars)")

            if len(part) > chunk_size:
                print(f"      Sentence {i+1}: TOO LONG ({len(part)} chars) -> HARD SPLITTING...")
                start = 0
                step = chunk_size - overlap
                while start < len(part):
                    piece = part[start:start + chunk_size].strip()
                    if piece:
                        chunks.append(piece)
                        print(f"        Hard-split piece [{start}:{start+chunk_size}] -> {len(piece)} chars")
                    start += step
                current = ""
            else:
                print(f"      Sentence {i+1}: \"{part[:60]}...\" -> NEW CHUNK START" if len(part) > 60 else f"      Sentence {i+1}: \"{part}\" -> NEW CHUNK START")
                current = part

    if current.strip():
        chunks.append(current.strip())
        print(f"      >>> FINAL CHUNK SAVED (length: {len(current.strip())} chars)")

    return chunks


def recursive_chunk(
    pages: List[Dict],
    chunk_size: int = 500,
    overlap: int = 100
) -> List[Dict]:
    """
    Recursive-like chunking that tries to preserve sentence groups.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[Dict] = []

    print("\n" + "=" * 70)
    print("RECURSIVE CHUNKING")
    print(f"  chunk_size = {chunk_size} chars")
    print(f"  overlap    = {overlap} chars")
    print(f"  Strategy   : Split on sentences first, hard-split only if needed")
    print("=" * 70)

    for page_data in pages:
        print(f"\n  Page {page_data['page']} | Total text length: {len(page_data['text'])} chars")
        print(f"  {'~' * 60}")

        split_chunks = recursive_split_text(
            text=page_data["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for idx, chunk_text in enumerate(split_chunks, start=1):
            chunk_id = f"{page_data['source']}_p{page_data['page']}_c{idx}_recursive"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "source": page_data["source"],
                    "page": page_data["page"],
                    "chunking_method": "recursive",
                }
            )

            print(f"\n    Chunk #{idx}")
            print(f"      ID      : {chunk_id}")
            print(f"      Length  : {len(chunk_text)} chars")
            print(f"      Preview : \"{chunk_text[:100]}...\"" if len(chunk_text) > 100 else f"      Preview : \"{chunk_text}\"")

    print(f"\n  TOTAL CHUNKS CREATED: {len(chunks)}")
    print("=" * 70)
    return chunks


def semantic_chunk(
    pages: List[Dict],
    chunk_size: int = 500
) -> List[Dict]:
    """
    Very simple Week 1 semantic-style chunking:
    - split into sentence-like units
    - merge nearby sentences while size allows
    This is not a true embedding-driven semantic chunker,
    but it is a good Week 1 approximation.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    chunks: List[Dict] = []

    print("\n" + "=" * 70)
    print("SEMANTIC CHUNKING (Week 1 Approximation)")
    print(f"  chunk_size  = {chunk_size} chars")
    print(f"  Strategy    : Group consecutive sentences until size limit")
    print(f"  No overlap  : Sentences are grouped, not split with overlap")
    print("=" * 70)

    for page_data in pages:
        text = page_data["text"]
        sentences = [s.strip() for s in text.split(".") if s.strip()]

        print(f"\n  Page {page_data['page']} | Total text length: {len(text)} chars")
        print(f"  Sentences found: {len(sentences)}")
        print(f"  {'~' * 60}")

        current = ""
        chunk_num = 1

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip() + "."
            candidate = f"{current} {sentence}".strip() if current else sentence

            if len(candidate) <= chunk_size:
                print(f"    Sentence {i+1}: \"{sentence[:50]}...\" -> ADDED (total: {len(candidate)} chars)" if len(sentence) > 50 else f"    Sentence {i+1}: \"{sentence}\" -> ADDED (total: {len(candidate)} chars)")
                current = candidate
            else:
                if current.strip():
                    chunk_id = f"{page_data['source']}_p{page_data['page']}_c{chunk_num}_semantic"
                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "text": current.strip(),
                            "source": page_data["source"],
                            "page": page_data["page"],
                            "chunking_method": "semantic",
                        }
                    )
                    print(f"    >>> CHUNK #{chunk_num} SAVED | {len(current.strip())} chars | \"{current.strip()[:80]}...\"")
                    chunk_num += 1
                print(f"    Sentence {i+1}: \"{sentence[:50]}...\" -> NEW CHUNK START" if len(sentence) > 50 else f"    Sentence {i+1}: \"{sentence}\" -> NEW CHUNK START")
                current = sentence

        if current.strip():
            chunk_id = f"{page_data['source']}_p{page_data['page']}_c{chunk_num}_semantic"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": current.strip(),
                    "source": page_data["source"],
                    "page": page_data["page"],
                    "chunking_method": "semantic",
                }
            )
            print(f"    >>> FINAL CHUNK #{chunk_num} SAVED | {len(current.strip())} chars | \"{current.strip()[:80]}...\"")

    print(f"\n  TOTAL CHUNKS CREATED: {len(chunks)}")
    print("=" * 70)
    return chunks
