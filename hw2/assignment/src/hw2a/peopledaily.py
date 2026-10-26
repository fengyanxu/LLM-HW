"""Parse PeopleDaily1998 into article records with title and body."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

from .common import ID_RE, PEOPLEDAILY_DIR, clean_title, parse_token


def parse_all():
    """Return one record per article, plus parsing statistics and anomalies."""
    grouped: dict[str, list[list[str]]] = defaultdict(list)
    stats = Counter()
    anomalies: list[dict] = []
    for path in sorted(PEOPLEDAILY_DIR.glob("1998??.txt")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_no, line in enumerate(handle, 1):
                stats["lines"] += 1
                fields = line.split()
                if not fields:
                    stats["blank_lines"] += 1
                    continue
                match = ID_RE.fullmatch(fields[0])
                if not match:
                    stats["bad_id"] += 1
                    anomalies.append({"file": path.name, "line": line_no, "field": fields[0], "kind": "bad_id"})
                    continue
                article_id = "-".join(match.groups()[:3])
                words = []
                for field in fields[1:]:
                    word = parse_token(field)
                    if word is not None:
                        words.append(word)
                if not words:
                    stats["empty_paragraphs"] += 1
                    continue
                grouped[article_id].append(words)

    records = []
    for article_id, paragraphs in sorted(grouped.items()):
        body = [word for paragraph in paragraphs[1:] for word in paragraph]
        if not body:
            stats["no_body"] += 1
            continue
        records.append(
            {
                "id": article_id,
                "date": article_id[:8],
                "page": article_id[9:11],
                "title": clean_title(paragraphs[0]),
                "title_words": paragraphs[0],
                "body": body,
                "paragraphs": len(paragraphs),
                "body_sha256": hashlib.sha256(" ".join(body).encode("utf-8")).hexdigest(),
            }
        )
    stats["articles"] = len(records)
    return records, dict(stats), anomalies
