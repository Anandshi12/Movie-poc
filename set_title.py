#!/usr/bin/env python3
"""
set_title.py - fix the movie title everywhere in the saved analysis WITHOUT calling any AI.

Use it when the title came out wrong (for example "Tangle" guessed from the file name instead of "Tangled").
It changes the title, the call to action and every narration line, then rewrites script/<stem>_script.txt.

    python3 src/set_title.py "Tangled"                  # replaces the old title (read from the analysis)
    python3 src/set_title.py "Tangled" --old "Tangle"

Then:  python3 src/narrate.py   (only the changed lines are re-voiced)
       python3 src/cut_clips.py (only the changed clips and the end card are rebuilt)
       python3 src/render_video.py
"""

import argparse
import json
import re
from pathlib import Path

from common import d, newest_analysis, stem_of


def main():
    ap = argparse.ArgumentParser(description="Change the movie title in the saved analysis.")
    ap.add_argument("title", help="The correct title, e.g. Tangled")
    ap.add_argument("--old", help="The wrong title to replace (default: the title stored in the analysis)")
    ap.add_argument("--analysis", type=Path)
    args = ap.parse_args()

    path = args.analysis or newest_analysis()
    data = json.loads(path.read_text(encoding="utf-8"))
    old = args.old or data.get("title", "")
    if not old:
        raise SystemExit("No old title found; pass --old.")
    pattern = re.compile(rf"(?<![A-Za-z]){re.escape(old)}(?![A-Za-z])", re.I)     # never matches inside 'Tangled'
    count = 0

    def fix(text):
        nonlocal count
        new, n = pattern.subn(args.title, text)
        count += n
        return new

    data["title"] = args.title
    data["cta"] = fix(data["cta"])
    for clip in data["clips"]:
        clip["narration"] = fix(clip["narration"])
    if args.title.lower() not in data["cta"].lower():
        data["cta"] = data["cta"].rstrip() + f" Watch {args.title}."
        print("The old title was not in the call to action, so 'Watch <title>.' was added to it.")
    backup = path.with_name(f"{stem_of(path)}_analysis.pre_title.json")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    script = d("script", create=True) / f"{stem_of(path)}_script.txt"
    script.write_text("\n\n".join([c["narration"].strip() for c in data["clips"]] + [data["cta"].strip()]) + "\n",
                      encoding="utf-8")
    print(f"Title is now '{args.title}' ({count} replacement(s) in the text). CTA: {data['cta']}")


if __name__ == "__main__":
    main()
