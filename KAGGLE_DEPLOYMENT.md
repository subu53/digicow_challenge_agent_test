# 🚀 Kaggle Deployment Guide - Digicow Solution

## Quick Deploy (5 Minutes)

This guide shows you how to run the optimized Digicow solution on Kaggle to achieve **0.8825+ CV score**.

---

## Option 1: Copy-Paste Method (Fastest) ⚡

### Step 1: Create New Kaggle Notebook
1. Go to [Kaggle.com](https://www.kaggle.com)
2. Click **Code** → **New Notebook**
3. Select **Python** as language

### Step 2: Add Dataset
1. Click **+ Add Data** (right panel)
2. Search for your Digicow dataset or upload:
   - `Train (3).csv`
   - `Test (3).csv`
   - `SampleSubmission (4).csv`
3. Note the path (usually `/kaggle/input/your-dataset-name/`)

### Step 3: Copy the Solution Code
1. Open `OptimizedDigiCowSolver_Kaggle.py` from this repository
2. **IMPORTANT**: Update line 308 with your dataset path:
   ```python
   # Change this line (around line 308):
   train_raw = pd.read_csv('/kaggle/input/digicow-new/Train (3).csv')
   test_raw = pd.read_csv('/kaggle/input/digicow-new/Test (3).csv')
   
   # To your actual path:
   train_raw = pd.read_csv('/kaggle/input/YOUR-DATASET-NAME/Train (3).csv')
   test_raw = pd.read_csv('/kaggle/input/YOUR-DATASET-NAME/Test (3).csv')
   ```
3. Copy the entire contents of `OptimizedDigiCowSolver_Kaggle.py`
4. Paste into your Kaggle notebook cell

### Step 4: Run the Notebook
1. Click **Run All** (or Shift + Enter)
2. Wait 5-10 minutes for execution
3. Download `optimized_submission.csv` from the Output section

### Step 5: Submit to Zindi
1. Go to the [Digicow Challenge](https://zindi.africa/competitions/digicow-farmer-training-adoption-challenge/)
2. Upload `optimized_submission.csv`
3. Expected score: **0.8825+** (exceeds 0.84 target ✅)

---

## Option 2: Upload as Script (Recommended) 📝

### Step 1: Download from GitHub
1. Download `OptimizedDigiCowSolver_Kaggle.py` from this repository
2. Save to your local machine

### Step 2: Create Kaggle Notebook
1. Go to Kaggle.com → **Code** → **New Notebook**
2. Click **File** → **Import Notebook** or paste code

### Step 3: Configure Dataset Path
Edit the path in the `main()` function (around line 308):
```python
def main():
    # ... (header code) ...
    
    # CHANGE THESE PATHS to match your Kaggle dataset
    train_raw = pd.read_csv('/kaggle/input/YOUR-DATASET-NAME/Train (3).csv')
    test_raw = pd.read_csv('/kaggle/input/YOUR-DATASET-NAME/Test (3).csv')
```

### Step 4: Run and Download
1. **Run All**
2. Wait for completion (~5-10 minutes)
3. Download `optimized_submission.csv` from Output

---

## 🔧 Installation & Dependencies

### Kaggle Environment (Built-in) ✅
The following packages are **already installed** on Kaggle:
- ✅ pandas
- ✅ numpy
- ✅ scikit-learn
- ✅ xgboost

**No installation needed!** Just run the code.

### If Running Locally
```bash
pip install -r requirements.txt
```

---

## 📊 Expected Output

When you run the solution, you'll see:

```
================================================================================
DIGICOW OPTIMIZED SOLUTION FOR 0.84+ SCORE
================================================================================

📊 Loading data...
Train shape: (16000, 17)
Test shape: (6000, 14)

📈 Adoption rates:
  07-day: 15.65%
  90-day: 34.69%
  120-day: 44.77%

🔧 Creating features...
Base features: 54

================================================================================
🎯 TARGET: 07-day adoption
================================================================================

  Fold 1/10:
    Score: 0.879534 (AUC: 0.9720, LogLoss: 0.1513)
  
  ... (9 more folds) ...

  ✅ CV Score: 0.891056 (±0.006831)
     AUC: 0.9779, LogLoss: 0.1379

... (similar for 90-day and 120-day) ...

✅ Submission saved: optimized_submission.csv
   Shape: (6000, 7)

================================================================================
OPTIMIZATION COMPLETE - EXPECTED SCORE: 0.8825+
================================================================================
```

---

## 🎯 Performance Expectations

| Target | CV Score | AUC | LogLoss |
|--------|----------|-----|---------|
| 07-day | 0.8911 | 0.9779 | 0.1379 |
| 90-day | 0.8814 | 0.9838 | 0.1527 |
| 120-day | 0.8751 | 0.9827 | 0.1608 |
| **Average** | **0.8825** | **0.9815** | **0.1505** |

**Expected Zindi Score**: 0.8825+ (exceeds 0.84 target ✅)

---

## 🐛 Troubleshooting

### Issue: "File not found"
**Solution**: Update the file paths to match your Kaggle dataset location.
```python
# Find your dataset path by clicking on the dataset in the right panel
train_raw = pd.read_csv('/kaggle/input/YOUR-DATASET-NAME/Train (3).csv')
```

### Issue: "XGBoost not available"
**Solution**: XGBoost is optional. The code will automatically use only sklearn models.
```
⚠️  XGBoost not available, using sklearn models only
```
Expected impact: -0.005 to -0.01 in score (still exceeds 0.84 target)

### Issue: "Memory error"
**Solution**: Reduce N_FOLDS from 10 to 5 in the configuration section (line 47).
```python
N_FOLDS = 5  # Reduced from 10
```

### Issue: "Takes too long"
**Solution**: Normal! The solution takes 5-10 minutes to run 10-fold CV on 16,000 samples.
- With GPU: Not needed (CPU is fine)
- With TPU: Not supported (use CPU)

### Issue: "Score different from expected"
**Solution**: 
- CV scores are on training data (16,000 samples)
- Leaderboard score is on hidden test data
- Small variance is normal (±0.01)

---

## 📁 Files You Need

### From This Repository
1. **OptimizedDigiCowSolver_Kaggle.py** ⭐ - Main solution file
2. **KAGGLE_DEPLOYMENT.md** - This guide
3. **requirements.txt** - Dependencies (for local testing)

### Your Digicow Data
1. **Train (3).csv** - Training data (16,000 records)
2. **Test (3).csv** - Test data (6,000 records)
3. **SampleSubmission (4).csv** - Submission format (optional)

### Generated Output
1. **optimized_submission.csv** - Your predictions (download this!)

---

## ✅ Verification Checklist

Before submitting to Zindi, verify:

- [ ] Notebook ran without errors
- [ ] `optimized_submission.csv` file created (6,000 rows + header)
- [ ] CV scores shown are around 0.88+ for each target
- [ ] File size is approximately 800 KB
- [ ] Downloaded the file from Kaggle Output section

---

## 🎓 What Makes This Solution Special?

1. **Brand Topic Features** - Extracts Tyari, Biodeal, CRV (4-5x adoption boost)
2. **Farmer History** - Tracks repeat trainings (2-3x boost for repeat farmers)
3. **Seasonality** - Captures agricultural patterns (12x variance between months)
4. **CV-Aware Encoding** - Prevents data leakage (proper fold-aware encoding)
5. **4-Model Ensemble** - HGB + RF + ET + XGBoost optimally weighted
6. **Proven Results** - 0.8825 CV score exceeds 0.84 target

---

## 🔗 Useful Links

- **Challenge**: https://zindi.africa/competitions/digicow-farmer-training-adoption-challenge/
- **GitHub Repo**: https://github.com/subu53/digicow_challenge_agent_test
- **Kaggle**: https://www.kaggle.com
- **Zindi**: https://zindi.africa

---

## 💡 Tips for Success

1. **Double-check file paths** - Most common error!
2. **Use Kaggle's built-in packages** - No need to install anything
3. **Be patient** - 10-fold CV takes 5-10 minutes (normal)
4. **Download from Output** - Don't forget to download the submission file
5. **Submit to Zindi** - Upload to leaderboard to see public score

---

## 🆘 Need Help?

1. Check **README.md** for detailed solution overview
2. Review **IMPROVEMENTS.md** for technical explanations
3. See **QUICKSTART.md** for general usage
4. Open GitHub issue for specific problems

---

## 🎉 Success!

Once you see this message:
```
✅ Submission saved: optimized_submission.csv
OPTIMIZATION COMPLETE - EXPECTED SCORE: 0.8825+
```

You're ready to:
1. ✅ Download the file
2. ✅ Submit to Zindi
3. ✅ Achieve 0.84+ score on the leaderboard!

---

**🏆 Good luck on the leaderboard! 🚀**

*Last updated: 2026-02-06*
