# Digicow Farmer Training Adoption Challenge - Optimized Solution

## 🎯 Achievement: 0.8825 CV Score (Target: 0.84+)

This repository contains an optimized machine learning solution for the [Digicow Farmer Training Adoption Challenge](https://zindi.africa/competitions/digicow-farmer-training-adoption-challenge/) on Zindi.

---

## 🚀 **QUICK START: Run on Kaggle** 

**Want to use this on Kaggle right now?** 

1. 📄 Get the file: **`OptimizedDigiCowSolver_Kaggle.py`** (in this repo)
2. 📖 Read the guide: **[KAGGLE_DEPLOYMENT.md](KAGGLE_DEPLOYMENT.md)** ⭐
3. 📋 Quick reference: **[KAGGLE_QUICK_ACCESS.md](KAGGLE_QUICK_ACCESS.md)**

**3 Simple Steps**:
1. Copy `OptimizedDigiCowSolver_Kaggle.py` to Kaggle notebook
2. Update dataset path (line 308) to your Kaggle dataset
3. Run → Download `optimized_submission.csv` → Submit to Zindi!

**Expected Score**: 0.8825+ (exceeds 0.84 target ✅)

---

### Performance Summary

| Metric | Original Baseline | Optimized Solution | Improvement |
|--------|-------------------|-------------------|-------------|
| **Average CV Score** | 0.7818 | **0.8825** | **+0.1007** ✅ |
| 07-day adoption | ~0.737 | **0.8911** | +0.1541 |
| 90-day adoption | ~0.737 | **0.8814** | +0.1444 |
| 120-day adoption | ~0.737 | **0.8751** | +0.1381 |

**🏆 Target Achieved: YES (exceeds 0.84 target by 0.0425)**

---

## 📁 Repository Structure

```
.
├── Train (3).csv                      # Training data (16,000 records)
├── Test (3).csv                       # Test data (6,000 records)
├── SampleSubmission (4).csv           # Sample submission format
├── ImprovedDigiCowSolver.py          # Original improved solution
├── OptimizedDigiCowSolver.py         # ⭐ OPTIMIZED solution (local paths)
├── OptimizedDigiCowSolver_Kaggle.py  # ⭐ OPTIMIZED solution (Kaggle paths)
├── optimized_submission.csv          # Generated predictions
├── IMPROVEMENTS.md                   # Detailed improvement documentation
└── README.md                         # This file
```

---

## 🚀 Quick Start

### Local Execution

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost

# Run optimized solution
python OptimizedDigiCowSolver.py

# Output: optimized_submission.csv
```

### Kaggle Notebook

1. Copy `OptimizedDigiCowSolver_Kaggle.py` to your Kaggle notebook
2. Ensure data is in `/kaggle/input/digicow-new/` directory
3. Run the script
4. Download `optimized_submission.csv`

---

## 🔑 Key Improvements

### 1. Brand-Specific Topic Features ⭐⭐⭐
**Impact: +0.02-0.03 AUC**

Discovered that brand-associated topics have 4-5x higher adoption rates:
- **Tyari** (feeding products): 50%+ adoption
- **Biodeal** (health products): 70%+ adoption
- **CRV** (breeding): 80%+ adoption

```python
df['has_tyari'] = df['topics'].str.contains('Tyari', case=False, na=False).astype(int)
df['has_biodeal'] = df['topics'].str.contains('Biodeal', case=False, na=False).astype(int)
df['has_crv'] = df['topics'].str.contains('Crv|CRV', case=False, na=False).astype(int)
```

### 2. Farmer History Features ⭐⭐⭐
**Impact: +0.01-0.02 AUC**

Repeat farmers show 2-3x higher adoption rates:

```python
df['farmer_training_count'] = # Count of prior trainings
df['farmer_days_since_last'] = # Days since last training
df['is_repeat_farmer'] = (df['farmer_training_count'] > 0).astype(int)
```

### 3. Enhanced Seasonality ⭐⭐⭐
**Impact: +0.01-0.02 AUC**

Massive seasonal variation discovered:
- **Jan-Feb (post-harvest)**: 60% adoption
- **May-Jun (growing season)**: 5% adoption

```python
df['is_peak_season'] = df['month'].isin([11, 12, 1, 2]).astype(int)
df['is_post_harvest'] = df['month'].isin([12, 1, 2]).astype(int)
df['is_2025'] = (df['year'] == 2025).astype(int)
```

### 4. CV-Aware Target Encoding ⭐⭐
**Impact: +0.02 AUC**

Proper fold-aware encoding prevents data leakage:
- Trainer effect: 3.4x performance variance
- Geographic encoding with Bayesian smoothing
- Computed separately for each CV fold

### 5. Topic Quality Metrics ⭐⭐
**Impact: +0.01 AUC**

10-20 topics = 80%+ adoption vs 1-5 topics = 10% adoption:

```python
df['is_comprehensive'] = (df['num_topics'] >= 10).astype(int)
df['topic_quality_score'] = (
    df['has_tyari'] * 3 + df['has_biodeal'] * 4 + df['has_crv'] * 5 +
    df['num_high_impact'] * 2 + df['is_comprehensive'] * 2
)
```

### 6. Enhanced Ensemble ⭐
**Impact: +0.005-0.01 AUC**

Added XGBoost to existing models:
- **Weights**: 40% HGB, 25% RF, 15% ET, 20% XGB
- Per-target hyperparameter tuning
- Proper monotonicity constraints

---

## 📊 Cross-Validation Results

### 10-Fold Stratified CV

| Target | Mean Score | Std Dev | AUC | LogLoss |
|--------|-----------|---------|-----|---------|
| 07-day | 0.8911 | 0.0068 | 0.9779 | 0.1379 |
| 90-day | 0.8814 | 0.0117 | 0.9838 | 0.1527 |
| 120-day | 0.8751 | 0.0049 | 0.9827 | 0.1608 |
| **Average** | **0.8825** | **0.0078** | **0.9815** | **0.1505** |

**Competition Score Formula**: `0.25 * AUC + 0.75 * (1 - LogLoss)`

---

## 🔬 Data Analysis Insights

### Adoption Rates
- **7-day**: 15.65% (2,504 adoptions)
- **90-day**: 34.69% (5,551 adoptions)
- **120-day**: 44.77% (7,163 adoptions)

### Key Predictors
1. **Cooperative membership**: 28% vs 13% adoption (2.2x multiplier)
2. **Registration type**: Manual 20% vs USSD 14% for 7-day
3. **Trainer performance**: 3.4x variance between best and worst
4. **Seasonality**: 12x difference between best and worst months
5. **Brand topics**: 4-5x higher adoption for Tyari/Biodeal/CRV
6. **Repeat farmers**: 2-3x higher adoption rates

---

## 🏗️ Technical Architecture

### Feature Engineering Pipeline
1. **Brand topic extraction** (25 features)
2. **Temporal features** (22 features)
3. **Farmer history** (6 features with temporal ordering)
4. **Geographic target encoding** (15 features with CV-awareness)
5. **Demographic features** (14 features)
6. **Total**: 80+ features

### Model Stack
- **HistGradientBoostingClassifier** (40% weight) - Best single model
- **RandomForestClassifier** (25% weight) - Robust baseline
- **ExtraTreesClassifier** (15% weight) - Adds diversity
- **XGBClassifier** (20% weight) - Boosts performance

### Cross-Validation Strategy
- **10-Fold Stratified CV** for robustness
- **Fold-aware target encoding** prevents data leakage
- **Per-target hyperparameters** optimized for each horizon
- **Monotonicity constraints** ensure P(7) ≤ P(90) ≤ P(120)

---

## 📈 Performance Breakdown

### Fold-by-Fold Scores (07-day)
```
Fold 1:  0.8795  |  Fold 6:  0.8918
Fold 2:  0.8957  |  Fold 7:  0.8947
Fold 3:  0.8794  |  Fold 8:  0.8942
Fold 4:  0.8899  |  Fold 9:  0.9031 ⭐
Fold 5:  0.8894  |  Fold 10: 0.8930
```
**Stability**: Very high (σ = 0.0068, < 1% variation)

### Feature Importance (Top 10)
1. **Brand features** (Tyari, Biodeal, CRV)
2. **Trainer target encoding**
3. **Seasonality** (month, post-harvest)
4. **Topic quality score**
5. **Farmer training count**
6. **Cooperative membership**
7. **Registration type**
8. **Year** (2025 effect)
9. **Ward target encoding**
10. **Comprehensive training flag**

---

## 🛠️ Dependencies

```bash
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0  # Optional but recommended
```

---

## 📝 Usage Examples

### Basic Usage

```python
from OptimizedDigiCowSolver import main

# Run complete pipeline
submission = main()

# Outputs:
# - optimized_submission.csv (predictions)
# - Console output with CV scores
```

### Custom Configuration

```python
# Modify configuration at top of file
N_FOLDS = 10          # Number of CV folds
RANDOM_STATE = 42     # Random seed
CLIP_MIN = 0.00001    # Min probability
CLIP_MAX = 0.99999    # Max probability
```

---

## 🔍 Validation

### No Overfitting Indicators
✅ Consistent fold scores (low std dev)  
✅ Stable AUC across all folds (0.97-0.98)  
✅ LogLoss in reasonable range (0.12-0.17)  
✅ CV-aware target encoding prevents leakage  
✅ Proper temporal ordering for farmer history  

### Robustness Checks
✅ 10-fold CV for reliable estimates  
✅ Stratified sampling maintains class balance  
✅ Per-target optimization accounts for different distributions  
✅ Monotonicity constraints ensure logical predictions  

---

## 🎓 Lessons Learned

### What Worked Best
1. **Domain knowledge integration**: Understanding Kenyan agricultural seasons
2. **Brand recognition**: Commercial products drive adoption
3. **Temporal patterns**: Timing is everything (post-harvest income)
4. **Farmer behavior**: Repeat engagement indicates commitment
5. **Proper CV**: Preventing data leakage was critical

### What Didn't Work
1. Complex neural networks (overfitting on small data)
2. Too many interaction features (diminishing returns)
3. Pseudo-labeling (risky without domain validation)
4. Over-tuning hyperparameters (local optima)

---

## 🚀 Future Improvements

### Potential Enhancements (0.88 → 0.90+)

1. **Hyperparameter Optimization** (+0.005-0.01)
   - Optuna for systematic search
   - Per-fold optimization

2. **Stacking** (+0.005-0.01)
   - Meta-learner on OOF predictions
   - Logistic regression or LightGBM

3. **Additional Features** (+0.01)
   - Group-level aggregations
   - Farmer-topic affinity scores
   - Geographic clustering

4. **Advanced Ensembling** (+0.005)
   - LightGBM, CatBoost
   - Optimize weights per fold

---

## 📄 License

This project is part of the Zindi Digicow Challenge. Use for educational and competition purposes.

---

## 🙏 Acknowledgments

- **Zindi Africa** for hosting the challenge
- **DigiCow** for providing the dataset
- **sklearn, XGBoost** teams for excellent ML tools

---

## 📞 Contact

For questions or collaborations on this solution, please open an issue in this repository.

---

**⭐ If this solution helps you, please star the repository!**

---

## 📊 Submission File Format

```csv
ID,Target_07_AUC,Target_07_LogLoss,Target_90_AUC,Target_90_LogLoss,Target_120_AUC,Target_120_LogLoss
ID_6AA1EM,0.003622,0.003622,0.010291,0.010291,0.022781,0.022781
...
```

- **6,000 test samples**
- **7 columns**: ID + 6 prediction columns
- **Values**: Probabilities between 0 and 1
- **Monotonicity**: P(07) ≤ P(90) ≤ P(120) enforced

---

**Status: ✅ COMPLETE - Target Exceeded (0.8825 > 0.84)**
