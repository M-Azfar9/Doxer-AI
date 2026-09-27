import json

nb = json.load(open("notebooks/docgen/3_docgen_sub_agent.ipynb", encoding="utf-8"))

def show_cell(idx):
    c = nb["cells"][idx]
    print(f"\n{'='*70}\nCELL {idx} ({c['cell_type']}):\n{'='*70}")
    src = "".join(c.get("source", []))
    print(src[:2000])

# Inspect key cells: State (3), connectors (7), router (11), subgraphs (16-22), synthesizer (24), critic (26), refinement (28), emitter (30), assembly (32)
for i in [2, 3, 5, 7, 10, 11, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]:
    show_cell(i)
