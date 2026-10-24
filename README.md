# BigMart Sales Prediction

Analytics Vidhya practice problem: predict `Item_Outlet_Sales` for item-outlet pairs.

- `bigmart_solution.py` - final model; reproduces `submission.csv` (CV RMSE ~1073, public score 1145.99)
- `graphs.ipynb` - EDA and model charts (PNGs saved in `plots/`)
- `data/` - competition train/test files
- `work/` - experiment scripts

Approach: clean missing values, predict the raw target (not log) from `Item_MRP` and `Outlet_Identifier`, blending a regularized gradient-boosting model with per-outlet piecewise-linear regression.

Run: `py bigmart_solution.py`
