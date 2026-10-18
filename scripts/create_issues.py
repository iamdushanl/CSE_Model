"""Create one GitHub issue per task in docs/tasks.csv (Teddy runs this once).

    python scripts/create_issues.py            # dry run: prints the commands
    python scripts/create_issues.py --run      # really creates them (needs the GitHub CLI: gh auth login)

Assign owners on GitHub afterwards (the owner column holds the work-package name, not a login).
"""
import csv
import subprocess
import sys

run = "--run" in sys.argv
with open("docs/tasks.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        title = f"[{row['id']}] {row['title']} (week {row['week']}, {row['owner']})"
        body = f"Owner: {row['owner']}\nWeek: {row['week']}\nSee handbook section 6 for outputs and the 'done when' criteria."
        cmd = ["gh", "issue", "create", "--title", title, "--body", body, "--label", row["label"]]
        if run:
            subprocess.run(cmd, check=False)
        else:
            print(" ".join(repr(c) if " " in c else c for c in cmd))
