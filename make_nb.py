import nbformat, re
src = open("analysis_src.py").read()
cells = []
for chunk in re.split(r"^# %%", src, flags=re.M)[1:]:
    if chunk.startswith(" [markdown]"):
        body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in chunk.split("\n")[1:]).strip()
        cells.append(nbformat.v4.new_markdown_cell(body))
    else:
        cells.append(nbformat.v4.new_code_cell(chunk.strip("\n")))
nb = nbformat.v4.new_notebook(); nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbformat.write(nb, "analysis.ipynb")
