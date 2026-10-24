"""Convert each stage's .py (cells split by '# %%') into an executed .ipynb next to it."""
import re, sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).parent
STAGES = {"01_eda": "eda", "02_cleaning": "cleaning",
          "03_feature_engineering": "feature_engineering", "04_modeling": "modeling"}

for folder, name in STAGES.items():
    src = (ROOT / folder / f"{name}.py").read_text(encoding="utf-8")
    cells = []
    for chunk in re.split(r"^# %%", src, flags=re.M)[1:]:
        md = chunk.startswith(" [markdown]")
        body = chunk.split("\n", 1)[1] if "\n" in chunk else ""
        if md:
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip().splitlines())
            cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            cells.append(nbformat.v4.new_code_cell(body.strip()))
    nb = nbformat.v4.new_notebook(cells=cells)
    NotebookClient(nb, timeout=300, resources={"metadata": {"path": str(ROOT / folder)}}).execute()
    nbformat.write(nb, ROOT / folder / f"{name}.ipynb")
    print("built", folder / Path(f"{name}.ipynb"))
