# BigMart Sales Prediction

Analytics Vidhya practice problem: predict `Item_Outlet_Sales` for item-outlet pairs.

- `bigmart_solution.py` - final model; reproduces `submission.csv` (CV RMSE ~1073, public score 1145.99)
- `graphs.ipynb` - EDA and model charts (PNGs saved in `plots/`)
- `data/` - competition train/test files
- `work/` - experiment scripts

Approach: clean missing values, predict the raw target (not log) from `Item_MRP` and `Outlet_Identifier`, blending a regularized gradient-boosting model with per-outlet piecewise-linear regression.

Run: `py bigmart_solution.py`

## Leaderboard journey

- **Initial rank: 203** -> **now: 306** (dropped as other participants improved).

## Two solutions, two scores

| Solution | Public RMSE | Verdict |
|---|---|---|
| `bigmart_solution.py` (raw target, `Item_MRP` + `Outlet_Identifier`, GBM blended with per-outlet piecewise-linear regression) | **1145.9888822358** | good score |
| `final_submission/solution.py` (sqrt target, more engineered features, blend of boosting models) | **1158.6890136233** | not a good score |

The second approach had a better-looking 5-fold CV (1084.6) but scored worse on the public board: the extra
features and boosting blend didn't generalise better than the simpler MRP x outlet model. The public split is
only 25% of test, so it is noisy as well.

Single-file version: `final_submission/solution.py` - the whole second solution in a single file (writes `final_submission/submission.csv`)
