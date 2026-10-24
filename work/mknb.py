import re, nbformat as nbf
src=open("graphs.py",encoding="utf-8-sig").read()
src=src.replace('B=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(B,"data"); O=os.path.join(B,"plots")','D="data"; O="plots"; os.makedirs(O,exist_ok=True)')
src=src.replace("matplotlib.use(\"Agg\")\n","%matplotlib inline\n").replace("import matplotlib\n","import matplotlib\n",1)
src=src.replace("def save(n): plt.tight_layout(); plt.savefig(os.path.join(O,n)); plt.close()","def save(n): plt.tight_layout(); plt.savefig(os.path.join(O,n)); plt.show()")
parts=re.split(r"\n(?=# \d )",src)
nb=nbf.v4.new_notebook(); c=[nbf.v4.new_markdown_cell("# BigMart Sales Prediction - EDA and model graphs\nRun from `D:\\AMBA` (needs `data/` folder). Each figure is displayed here and saved as a PNG in `plots/`.")]
head=parts[0].replace("import os, numpy","import os, numpy")
c.append(nbf.v4.new_code_cell(head.strip()))
titles={"1":"Target distribution","2":"Sales vs MRP","3":"Mean sales by outlet","4":"Sales by outlet type","5":"Missing values","6":"Model comparison (CV RMSE from experiments)","7":"Out-of-fold predicted vs actual (final model)","8":"Per-outlet fits"}
for p in parts[1:]:
    n=p[2:3]; c.append(nbf.v4.new_markdown_cell("## "+titles[n])); c.append(nbf.v4.new_code_cell(p.strip()))
nb.cells=c; nbf.write(nb,"graphs.ipynb")

