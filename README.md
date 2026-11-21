# PH-Soil-Prediction

A Streamlit app for predicting soil pH from soil properties and nutrient measurements. You upload your own labelled data, train a regression model, check its error, then predict pH for new samples (single entry or batch file) and get a simple indicative recommendation.

Companion tool to the study on interpretable machine learning for digital soil pH mapping:

> Alkaf ZZ et al. (2026) Interpretable machine learning for digital soil pH mapping using an optimized AdaBoost algorithm. *TELKOMNIKA* 24(4). https://doi.org/10.12928/telkomnika.v24i4.27724

Note: the optimized AdaBoost model reported in the paper is not part of this app. The app offers ten general-purpose regressors for exploration (see below).

## Features

- Upload an Excel training file and train one of ten regressors: KNN, Decision Tree, Random Forest, SVR, Gradient Boosting, Neural Network (MLP), XGBoost, LightGBM, Polynomial Regression, Bayesian Ridge
- Hold-out evaluation (80/20 split): R², MSE, and RMSE, with an actual-versus-predicted plot
- Single-sample prediction with an error margin (RMSE), or batch prediction from an Excel/CSV file with CSV download
- Indicative pH class: acidic (below 5.5), neutral (5.5 to 7.5), alkaline (above 7.5)

## Data format

The training file is an Excel workbook with a sheet named `SOIL DATA`. It needs a column named `pH` (target) and numeric feature columns, for example: Sand %, Clay %, Silt %, EC, organic matter, CaCO3, N-NO3, P, K, Mg, Fe, Zn, Mn, Cu, B. An `ID` column, if present, is ignored. Completely empty columns are dropped. Missing values are filled with the column mean.

**No dataset is included in this repository.** Use your own data, or obtain the data described in the paper.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limits

- The recommendations (liming for acidic soil, sulfur for alkaline soil) are indicative only and are not agronomic advice. Confirm with a local soil test and an agronomist.
- Models are trained in the session and are not saved between runs.

## License

MIT, see [LICENSE](LICENSE).
