import json

nb = json.load(open("notebooks/docgen/3_docgen_sub_agent.ipynb", encoding="utf-8"))
print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb["cells"]):
    cell_type = cell.get("cell_type", "")
    src = "".join(cell.get("source", []))
    first_line = src.strip().split("\n")[0] if src.strip() else "(empty)"
    print(f"Cell {i:02d} [{cell_type}]: {first_line[:90]}")
