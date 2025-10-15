# 🪐 Exoplanet Classifier

**Identifying confirmed exoplanets from NASA Kepler telescope detections using machine learning.**

When the Kepler Space Telescope detects a periodic dimming of a star, it flags a "Kepler Object of Interest" (KOI) — a candidate exoplanet. But not all KOIs are real planets. Some are false positives caused by binary stars, instrument noise, or other astrophysical phenomena.

This project builds a machine learning classifier to distinguish **confirmed exoplanets** from **false positives**, achieving **96% accuracy** and helping prioritize which candidates deserve costly follow-up observations.

---

## 🎯 Problem Statement

**Challenge:** Out of 10,000+ Kepler detections, only ~2,600 are confirmed exoplanets. Manual verification is:
- **Time-consuming**: Requires telescope time and expert analysis
- **Expensive**: Follow-up observations cost thousands per target
- **Error-prone**: Human judgment on ambiguous signals

**Solution:** Automated classification using 74 astrophysical features (orbital parameters, stellar properties, transit characteristics) to flag high-confidence candidates.

---

## 📊 Dataset

- **Source**: [NASA Exoplanet Archive](https://exoplanetarchive.ipas.caltech.edu/) - Kepler KOI Cumulative Table
- **Original size**: 9,564 Kepler Objects of Interest
- **After preprocessing**: 6,385 samples (CONFIRMED: 2,675 | FALSE POSITIVE: 3,710)
- **Features**: 74 astrophysical parameters including:
  - Transit properties (depth, duration, period)
  - Orbital mechanics (semi-major axis, eccentricity, inclination)
  - Stellar characteristics (temperature, radius, metallicity)
  - Signal quality metrics (SNR, number of transits)

**Key challenges:**
- ⚠️ **Imbalanced classes** (58% false positives vs 42% confirmed)
- ⚠️ **31% rows with missing values** across 107 features
- ⚠️ **High-dimensional feature space** with multicollinearity

---

## 🔧 Data Preprocessing

Systematic approach to handle messy astronomical data:

### 1. Feature Selection
- ❌ Removed metadata (IDs, comments, data provenance)
- ❌ Removed leakage features (disposition flags known only post-verification)
- ❌ Dropped 12 DICCO/DIKCO features (centroid analysis) with >50% missing values
- ❌ Removed constant features (e.g., `koi_eccen` = 0 for all samples)
- ✅ **Result**: 107 → 74 features

### 2. Missing Data Strategy
Rather than impute uncertain astrophysical measurements:
- Dropped columns with 100% missing (no information)
- Dropped columns with >15% missing if not critical
- Dropped remaining rows with NaNs (reduced samples by 18%)
- **Rationale**: Preserves data integrity over quantity

### 3. Target Encoding
- Excluded "CANDIDATE" class (ambiguous, not verified)
- Binary classification: `CONFIRMED` vs `FALSE POSITIVE`

**Final dataset**: 6,385 samples × 74 features (no missing values)

---

## 🤖 Model Development

### Baseline Models (No Hyperparameter Tuning)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 88.4% | 88.6% | 88.4% | 88.5% | 0.94 |
| Random Forest | 95.9% | 96.0% | 95.9% | 95.9% | 0.99 |
| **XGBoost** | **98.0%** | **98.0%** | **98.0%** | **98.0%** | **0.99** |

**Key observations:**
- Logistic Regression struggles with non-linear patterns (as expected from EDA)
- Tree-based models excel due to complex feature interactions
- XGBoost achieves near-perfect separation without hyperparameter tuning

### Test Set Performance (Final Evaluation)

**XGBoost Final Metrics:**
- Accuracy: **96.2%**
- Precision: **95.6%** (few false alarms)
- Recall: **95.6%** (catches most real exoplanets)
- ROC-AUC: **0.992** (excellent class separation)

**No overfitting observed** (train/test performance similar), indicating the model generalizes well.

---

## 📈 Feature Importance

Top predictors (account for 87% of total importance):

| Feature | Importance | Description |
|---------|-----------|-------------|
| `koi_smet_err2` | 41% | Stellar metallicity uncertainty |
| `koi_prad_err2` | 8% | Planet radius uncertainty |
| `koi_dor` | 6% | Distance over stellar radius |
| `koi_prad` | 5% | Planet radius |
| `koi_max_mult_ev` | 3% | Maximum multiple event statistic |

**Insights:**
- **Measurement uncertainty** is the strongest signal (higher uncertainty → likely false positive)
- **Orbital geometry** (`koi_dor`) distinguishes eclipsing binaries from planets
- **Planet size** matters: extremely large "planets" are often misclassified stellar companions

---

## 🚀 Demo

![Streamlit Interface](image.png)


Input custom KOI parameters to get instant predictions.

---

## 💻 Installation & Usage

### Local Setup

```bash
git clone https://github.com/TheCoolerMendeiros/exoplanet-classifier
cd exoplanet-classifier
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run app.py
```

### Use Model Directly

```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load('exoplanet_xgb_model.pkl')

# Load feature medians for missing value handling
medians = pd.read_csv('feature_medians.csv', index_col=0)

# Make prediction
prediction = model.predict(features)
```

### Jupyter Notebook

Open `notebooks/exoplanet_analysis.ipynb` for full analysis and model training walkthrough.

---

## 📁 Project Structure

```
exoplanet-classifier/
├── data/
│   └── cumulative_2025.10.04_16.38.22.csv  # NASA Kepler data
├── notebooks/
│   └── exoplanet_analysis.ipynb            # Full analysis
├── app.py                                   # Streamlit web app
├── exoplanet_xgb_model.pkl                 # Trained XGBoost model
├── feature_medians.csv                     # For missing value imputation
├── requirements.txt
├── image.png                               # Demo screenshot
└── README.md
```

---

## 🛠️ Tech Stack

- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, XGBoost
- **Deployment**: Streamlit
- **Domain Knowledge**: Astrophysics (Kepler mission data interpretation)

---

## 🔬 Key Takeaways

1. ✅ **Domain expertise matters**: Understanding astrophysical measurement uncertainties guided feature selection
2. ✅ **Data quality > quantity**: Dropping ambiguous samples improved model reliability
3. ✅ **Simple models win**: Default XGBoost revealed great performance
4. ✅ **Interpretability crucial**: Feature importance validates known astrophysical relationships
5. ⚠️ **Imbalanced data handled naturally**: Tree ensembles robust to 58/42 class split without SMOTE

---

## 📚 References

- [NASA Exoplanet Archive](https://exoplanetarchive.ipas.caltech.edu/)
- [Kepler Mission Overview](https://www.nasa.gov/mission_pages/kepler/main/index.html)
- [Thompson et al. (2018) - Kepler Data Release Notes](https://ui.adsabs.harvard.edu/abs/2018ApJS..235...38T)

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 🤝 Contact

Built by [Pedro Mendeiros] | [GitHub](https://github.com/TheCoolerMendeiros) | [LinkedIn](https://www.linkedin.com/in/pedro-mendeiros-159a801a8/)

*Interested in data science for astronomy or ML for scientific discovery? Let's connect!*
