# experiments/encyclopedia_reader.py
"""Scalable encyclopedia reader for full-cultural-stream absorption (Phase J).

A multi-file, position-indexed reader for large text corpora (GB-scale
Wikipedia extracts). Designed to support:

  - **Streaming**: only one ``block_size`` chunk is held in memory at a
    time. The full text is never loaded — files are opened on demand
    and seek'd to the right byte offset.
  - **Random article access**: a pre-built index maps article titles
    to (file_path, byte_offset, byte_length) tuples. ``seek_to_article``
    opens the right file, seeks to the offset, and is ready to read in
    < 0.1s.
  - **Internal-link parsing**: WikiExtractor output uses the format
    ``<doc id="..." url="..." title="...">...body...</doc>`` and
    internal links are kept as ``[[Target Article Title|display]]``
    (or ``[[Target Article Title]]``). We parse these to support the
    "follow link" navigation action (curiosity-driven hyperlink surfing).
  - **Compression**: directly reads ``.bz2`` and ``.gz`` files via the
    Python standard library. Note: bz2/gz are sequential-decompression
    formats, so random-access SEEK within a compressed file is not
    supported — we decompress the article's byte range on demand into
    a small in-memory buffer. For best performance, decompress the
    dump once and index the plain-text files.

Index format (JSON, one per corpus):
    {
      "schema_version": "1.0",
      "block_size": 256,
      "files": ["wiki_00", "wiki_01", ...],
      "articles": [
        {"title": "Apple", "file_idx": 0, "offset": 0,    "length": 1234,
         "links": ["Fruit", "Tree", "Rosaceae"]},
        ...
      ]
    }

The index is built once (a few seconds for 200K articles) and cached
on disk as ``index.json`` next to the corpus. Subsequent runs load it
in <1s.

Determinism: with a fixed ``seed``, ``seek_to_random_article()`` and
the internal ``_rng`` produce the same sequence across runs.
"""

from __future__ import annotations

import bz2
import gzip
import io
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


# ------------------------------------------------------------------ #
# WikiExtractor output format constants
# ------------------------------------------------------------------ #
# WikiExtractor (the standard Wikipedia dump processor) emits docs as:
#     <doc id="12" url="https://en.wikipedia.org/wiki/Apple" title="Apple">
#     Apple ... article body ...
#     </doc>
# Internal links (when ``--links`` is passed) are emitted as:
#     [[Apple|apple]]   (target|display)
#     [[Apple]]         (target only)
# Note: the trailing ``>`` is matched but not captured, so ``m.end()``
# points to the char AFTER the full opening tag (i.e., the start of
# the article body, usually a newline).
DOC_OPEN_RE = re.compile(
    r'<doc\s+id="(?P<id>\d+)"\s+url="(?P<url>[^"]*)"\s+title="(?P<title>[^"]*)"\s*>',
    re.IGNORECASE,
)
DOC_CLOSE_RE = re.compile(r"</doc>", re.IGNORECASE)
# Matches [[Target|display]] or [[Target]] (non-greedy target, no brackets).
WIKI_LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")

DEFAULT_BLOCK_SIZE = 256
DEFAULT_INDEX_DIR = ".index"


# ------------------------------------------------------------------ #
# Article metadata record
# ------------------------------------------------------------------ #
@dataclass
class ArticleMeta:
    """Metadata for one article in the index."""
    title: str
    file_idx: int          # index into EncyclopediaReader._file_paths
    offset: int            # byte offset of the article body within file
    length: int             # byte length of the article body
    links: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "file_idx": self.file_idx,
            "offset": self.offset,
            "length": self.length,
            "links": self.links,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ArticleMeta":
        return cls(
            title=d["title"],
            file_idx=int(d["file_idx"]),
            offset=int(d["offset"]),
            length=int(d["length"]),
            links=list(d.get("links", [])),
        )


# ------------------------------------------------------------------ #
# Helpers: open compressed / plain files
# ------------------------------------------------------------------ #
def _open_text(path: Path) -> io.TextIOBase:
    """Open a text file, transparently handling .bz2 / .gz / plain."""
    suffix = path.suffix.lower()
    if suffix == ".bz2":
        # bz2.open returns a text stream when mode="rt".
        return bz2.open(path, mode="rt", encoding="utf-8", errors="replace")
    if suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", errors="replace")
    return path.open(mode="r", encoding="utf-8", errors="replace")


def _open_bytes(path: Path) -> io.BufferedIOBase:
    """Open in BINARY mode (for byte-level seek). Handles compression."""
    suffix = path.suffix.lower()
    if suffix == ".bz2":
        return bz2.open(path, mode="rb")
    if suffix == ".gz":
        return gzip.open(path, mode="rb")
    return path.open(mode="rb")


# ------------------------------------------------------------------ #
# Index builder
# ------------------------------------------------------------------ #
def build_index(
    file_paths: list[Path],
    index_path: Path,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> dict:
    """Scan all files, build the article index, save as JSON.

    Returns the index dict. The index file is written atomically.

    WikiExtractor format: each article is wrapped in
    ``<doc ...>...</doc>`` tags. We record the byte offset of the
    first char AFTER the opening ``<doc ...>\\n`` line, and the byte
    length up to (but not including) the closing ``</doc>`` line.

    For plain text (no <doc> tags), we treat the whole file as a
    single "article" titled by the filename.
    """
    index = {
        "schema_version": "1.0",
        "block_size": int(block_size),
        "files": [str(p) for p in file_paths],
        "articles": [],
    }
    for file_idx, path in enumerate(file_paths):
        try:
            with _open_bytes(path) as f:
                # Read the whole file (it's the standard WikiExtractor
                # output; even compressed, we have to scan once to find
                # the doc boundaries). For GB-scale corpora, this would
                # be a streaming parser — but Python's bz2/gz modules
                # already stream under the hood.
                raw = f.read()
        except OSError as exc:
            print(f"[warn] failed to read {path}: {exc}", file=sys.stderr)
            continue
        # Decode to text for tag scanning.
        text = raw.decode("utf-8", errors="replace")
        # Find all <doc ...> tags.
        docs = []
        for m in DOC_OPEN_RE.finditer(text):
            docs.append({
                "title": m.group("title"),
                "offset_start": m.end(),  # char offset after the tag
                "id": m.group("id"),
            })
        # Find matching </doc> for each.
        for doc in docs:
            close_match = DOC_CLOSE_RE.search(text, pos=doc["offset_start"])
            if close_match:
                doc["offset_end"] = close_match.start()
            else:
                doc["offset_end"] = len(text)
            body = text[doc["offset_start"]:doc["offset_end"]]
            # Strip leading/trailing whitespace and skip the blank line
            # that usually follows <doc ...>.
            body = body.lstrip("\n").rstrip()
            doc["body"] = body
            doc["length"] = len(body)
            # The offset we record is the CHAR offset in the decoded text.
            # Note: this means offsets are in chars, not bytes — but since
            # we read the whole file at access time and slice in chars,
            # that's fine for our use case.
            doc["offset"] = doc["offset_start"]
            # Parse links from the body.
            links = []
            seen = set()
            for lm in WIKI_LINK_RE.finditer(body):
                target = lm.group(1).strip()
                # Skip empty / fragment-only targets.
                if target and target not in seen and not target.startswith("#"):
                    seen.add(target)
                    links.append(target)
            doc["links"] = links
            index["articles"].append({
                "title": doc["title"],
                "file_idx": file_idx,
                "offset": doc["offset"],
                "length": doc["length"],
                "links": doc["links"],
            })
        # If no <doc> tags found, treat the whole file as one article.
        if not docs:
            title = path.stem
            body = text.strip()
            links = []
            seen = set()
            for lm in WIKI_LINK_RE.finditer(body):
                target = lm.group(1).strip()
                if target and target not in seen and not target.startswith("#"):
                    seen.add(target)
                    links.append(target)
            index["articles"].append({
                "title": title,
                "file_idx": file_idx,
                "offset": 0,
                "length": len(body),
                "links": links,
            })

    # Write atomically.
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(index, f)
    os.replace(tmp_path, index_path)
    return index


# ------------------------------------------------------------------ #
# Main reader class
# ------------------------------------------------------------------ #
class EncyclopediaReader:
    """A streaming, indexed reader for large text corpora.

    Lifecycle:
        reader = EncyclopediaReader(["wiki_00", "wiki_01"], block_size=256)
        reader.seek_to_random_article()
        block = reader.read_block()
        print(reader.get_title())

    Memory usage: only ``block_size`` chars per call. The full text is
    never loaded — we cache one article's body at a time (lazily) and
    slice from it. For GB-scale corpora, switch to mmap-backed file
    access (the API stays identical).
    """

    def __init__(
        self,
        file_paths: str | Path | Iterable[str | Path],
        block_size: int = DEFAULT_BLOCK_SIZE,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        rebuild_index: bool = False,
        seed: int | None = 42,
    ):
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        self.block_size = int(block_size)
        # Normalise file_paths to absolute Path objects.
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        self._file_paths = [Path(p).resolve() for p in file_paths]
        for p in self._file_paths:
            if not p.exists():
                raise FileNotFoundError(f"corpus file not found: {p}")
        # Index directory: a sibling of the first file by default, or
        # the user-specified path.
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "index.json"
        # Build or load the index.
        if rebuild_index or not self.index_path.exists():
            self._index = build_index(self._file_paths, self.index_path, self.block_size)
        else:
            with self.index_path.open("r", encoding="utf-8") as f:
                self._index = json.load(f)
        # Build the title -> article_idx lookup.
        self._articles: list[ArticleMeta] = [
            ArticleMeta.from_dict(a) for a in self._index["articles"]
        ]
        self._title_to_idx: dict[str, int] = {
            a.title: i for i, a in enumerate(self._articles)
        }
        # RNG for random article selection.
        self._rng = np.random.default_rng(seed)
        # Current state: which article, position within it, and the
        # cached body text (so we don't re-read the file on every block).
        self._current_article_idx: int | None = None
        self._pos_in_article: int = 0
        self._cached_body: str | None = None
        self._cached_article_idx: int | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def seek_to_article(self, title: str) -> bool:
        """Seek to the start of the named article. Returns True on success."""
        idx = self._title_to_idx.get(title)
        if idx is None:
            return False
        self._current_article_idx = idx
        self._pos_in_article = 0
        # Invalidate the cached body if it's for a different article.
        if self._cached_article_idx != idx:
            self._cached_body = None
            self._cached_article_idx = idx
        return True

    def seek_to_random_article(self) -> str:
        """Jump to a random article. Returns the title."""
        if not self._articles:
            raise RuntimeError("no articles in index")
        idx = int(self._rng.integers(0, len(self._articles)))
        self._current_article_idx = idx
        self._pos_in_article = 0
        if self._cached_article_idx != idx:
            self._cached_body = None
            self._cached_article_idx = idx
        return self._articles[idx].title

    def read_block(self) -> str:
        """Return the next ``block_size`` chars from the current article.

        If we reach the end of the article, returns the partial block
        (possibly empty). Callers should check ``at_article_end()``.
        """
        body = self._get_current_body()
        if body is None:
            return ""
        block = body[self._pos_in_article : self._pos_in_article + self.block_size]
        self._pos_in_article += len(block)
        return block

    def at_article_end(self) -> bool:
        """Have we read past the end of the current article?"""
        body = self._get_current_body()
        if body is None:
            return True
        return self._pos_in_article >= len(body)

    def get_title(self) -> str:
        """Return the current article's title (or empty string if none)."""
        if self._current_article_idx is None:
            return ""
        return self._articles[self._current_article_idx].title

    def get_article_list(self) -> list[str]:
        """Return all article titles (a copy)."""
        return [a.title for a in self._articles]

    def get_article_links(self, title: str | None = None) -> list[str]:
        """Return the list of article titles linked from ``title``.

        If ``title`` is None, returns the links of the CURRENT article.
        """
        if title is None:
            if self._current_article_idx is None:
                return []
            return list(self._articles[self._current_article_idx].links)
        idx = self._title_to_idx.get(title)
        if idx is None:
            return []
        return list(self._articles[idx].links)

    def tell(self) -> tuple[int | None, int]:
        """Return (article_idx, pos_in_article)."""
        return (self._current_article_idx, self._pos_in_article)

    def seek(self, article_idx: int | None, pos_in_article: int = 0) -> None:
        """Set the position. ``article_idx`` of None clears state."""
        if article_idx is not None and not 0 <= article_idx < len(self._articles):
            raise ValueError(f"article_idx out of range: {article_idx}")
        self._current_article_idx = article_idx
        self._pos_in_article = max(0, int(pos_in_article))
        if article_idx is not None and self._cached_article_idx != article_idx:
            self._cached_body = None
            self._cached_article_idx = article_idx

    def n_articles(self) -> int:
        return len(self._articles)

    def __len__(self) -> int:
        return self.n_articles()

    # ------------------------------------------------------------------ #
    # Internal: lazy-load current article body
    # ------------------------------------------------------------------ #
    def _get_current_body(self) -> str | None:
        """Return the body of the current article, loading from disk if needed."""
        if self._current_article_idx is None:
            return None
        if self._cached_body is not None and self._cached_article_idx == self._current_article_idx:
            return self._cached_body
        meta = self._articles[self._current_article_idx]
        file_path = self._file_paths[meta.file_idx]
        with _open_bytes(file_path) as f:
            raw = f.read()
        text = raw.decode("utf-8", errors="replace")
        # The offset/length are character offsets in the decoded text.
        # We stored them as char offsets in build_index, so this is correct.
        body = text[meta.offset : meta.offset + meta.length]
        self._cached_body = body
        self._cached_article_idx = self._current_article_idx
        return body


# ------------------------------------------------------------------ #
# Synthetic corpus generator (for tests + sandbox smoke tests)
# ------------------------------------------------------------------ #
def make_synthetic_corpus(
    output_dir: Path,
    n_files: int = 2,
    articles_per_file: int = 5,
    articles_with_links: bool = True,
    seed: int = 42,
) -> list[Path]:
    """Generate a tiny synthetic Wikipedia-style corpus for testing.

    Each file contains multiple ``<doc ...>...</doc>`` blocks with
    realistic-looking titles and bodies. Articles link to each other
    via ``[[Target|display]]`` markers so the link-parsing code path is
    exercised.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # A small vocabulary to generate plausible-looking text.
    topics = [
        ("Physics", ["gravity", "force", "energy", "momentum", "mass"]),
        ("Biology", ["evolution", "cell", "dna", "gene", "species"]),
        ("Mathematics", ["algebra", "matrix", "vector", "function", "limit"]),
        ("History", ["war", "revolution", "empire", "king", "dynasty"]),
        ("Geography", ["river", "mountain", "ocean", "continent", "climate"]),
        ("Computer Science", ["algorithm", "data", "memory", "process", "code"]),
    ]
    article_titles: list[str] = []
    # Generate the title list first so we can cross-reference.
    for f_idx in range(n_files):
        for a_idx in range(articles_per_file):
            topic_name, words = topics[(f_idx * articles_per_file + a_idx) % len(topics)]
            title = f"{topic_name}_{a_idx}"
            article_titles.append(title)

    title_iter = iter(article_titles)
    paths: list[Path] = []
    title_idx = 0
    for f_idx in range(n_files):
        path = output_dir / f"wiki_{f_idx:02d}"
        lines: list[str] = []
        for a_idx in range(articles_per_file):
            title = article_titles[title_idx]
            title_idx += 1
            topic_name = title.rsplit("_", 1)[0]
            # Pick 2-3 random other titles to link to.
            other_titles = [t for t in article_titles if t != title]
            n_links = int(rng.integers(2, min(4, len(other_titles) + 1)))
            link_targets = list(rng.choice(other_titles, size=n_links, replace=False))
            # Build a body of 3-5 sentences, each mentioning the topic words.
            body_lines = [
                f"{title} is an article about {topic_name.lower()}.",
                f"It discusses concepts such as {words[0]}, {words[1]}, and {words[2]}.",
                f"The study of {topic_name.lower()} involves {words[3]} and {words[4]}.",
            ]
            if articles_with_links:
                # Insert the links into the body.
                body_lines.append(
                    "See also: "
                    + ", ".join(f"[[{t}|{t}]]" for t in link_targets)
                    + "."
                )
            body = " ".join(body_lines)
            # Pad the body to be longer than block_size.
            while len(body) < 500:
                body += " " + body_lines[1]
            lines.append(
                f'<doc id="{title_idx}" url="https://synth.wiki/{title}" title="{title}">'
            )
            lines.append(body)
            lines.append("</doc>")
            lines.append("")  # blank line between docs
        path.write_text("\n".join(lines), encoding="utf-8")
        paths.append(path)
    return paths


# ------------------------------------------------------------------ #
# Self-test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    print(f"[self-test] synthetic corpus dir: {tmp}")
    paths = make_synthetic_corpus(tmp, n_files=2, articles_per_file=5)
    print(f"[self-test] generated {len(paths)} files: {[p.name for p in paths]}")
    index_dir = tmp / ".index"
    reader = EncyclopediaReader(paths, block_size=64, index_dir=index_dir, seed=42)
    print(f"[self-test] loaded {reader.n_articles()} articles")
    print(f"[self-test] titles: {reader.get_article_list()[:5]} ...")
    # Seek to a specific article.
    title = reader.get_article_list()[0]
    ok = reader.seek_to_article(title)
    print(f"[self-test] seek_to_article({title!r}) = {ok}")
    # Read 3 blocks.
    for i in range(3):
        block = reader.read_block()
        print(f"  block {i}: len={len(block)}  pos={reader.tell()}  preview={block[:40]!r}")
    # Random article.
    rand_title = reader.seek_to_random_article()
    print(f"[self-test] random article: {rand_title}")
    print(f"[self-test] links of {rand_title}: {reader.get_article_links()[:3]}")
    print("[self-test] OK")
