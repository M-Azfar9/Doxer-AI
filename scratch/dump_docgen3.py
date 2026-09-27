import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

nb = json.load(open("notebooks/docgen/3_docgen_sub_agent.ipynb", encoding="utf-8"))

with open("scratch/docgen3_dump.txt", "w", encoding="utf-8") as out:
    for idx, c in enumerate(nb["cells"]):
        out.write(f"\n{'='*70}\nCELL {idx} ({c['cell_type']}):\n{'='*70}\n")
        src = "".join(c.get("source", []))
        out.write(src + "\n")

print("Dumped all 35 cells to scratch/docgen3_dump.txt")
