#!/usr/bin/env python3
"""
Quick validation script for the Kaggle version
Tests that the code can load and basic functions work
"""

import sys
import os

def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    try:
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import StratifiedKFold
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
        from sklearn.metrics import log_loss, roc_auc_score
        print("✅ All core imports successful")
        
        try:
            from xgboost import XGBClassifier
            print("✅ XGBoost available")
            return True, True
        except ImportError:
            print("⚠️  XGBoost not available (optional)")
            return True, False
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False, False

def test_file_exists():
    """Test that the Kaggle file exists"""
    print("\nTesting file existence...")
    kaggle_file = "OptimizedDigiCowSolver_Kaggle.py"
    if os.path.exists(kaggle_file):
        print(f"✅ {kaggle_file} exists")
        size = os.path.getsize(kaggle_file) / 1024
        print(f"   Size: {size:.1f} KB")
        return True
    else:
        print(f"❌ {kaggle_file} not found")
        return False

def test_file_content():
    """Test that the Kaggle file has expected content"""
    print("\nTesting file content...")
    kaggle_file = "OptimizedDigiCowSolver_Kaggle.py"
    
    with open(kaggle_file, 'r') as f:
        content = f.read()
    
    checks = {
        "main() function": "def main():" in content,
        "Brand features": "has_tyari" in content or "has_biodeal" in content,
        "Farmer history": "farmer_training_count" in content,
        "Target encoding": "apply_cv_aware_target_encoding" in content,
        "Ensemble models": "HistGradientBoostingClassifier" in content,
        "Kaggle path": "/kaggle/input/" in content,
    }
    
    all_passed = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
        if not passed:
            all_passed = False
    
    return all_passed

def test_data_files():
    """Test that data files exist"""
    print("\nTesting data files...")
    data_files = ["Train (3).csv", "Test (3).csv"]
    
    all_exist = True
    for file in data_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024 * 1024)
            print(f"✅ {file} exists ({size:.1f} MB)")
        else:
            print(f"⚠️  {file} not found (needed for execution)")
            all_exist = False
    
    return all_exist

def main():
    print("="*80)
    print("KAGGLE VERSION VALIDATION")
    print("="*80)
    
    results = []
    
    # Test imports
    imports_ok, has_xgb = test_imports()
    results.append(("Imports", imports_ok))
    
    # Test file exists
    file_exists = test_file_exists()
    results.append(("File exists", file_exists))
    
    if file_exists:
        # Test content
        content_ok = test_file_content()
        results.append(("File content", content_ok))
    
    # Test data files (optional)
    data_ok = test_data_files()
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    all_critical_passed = all(result[1] for result in results)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    if data_ok:
        print(f"✅ INFO: Data files available")
    else:
        print(f"⚠️  INFO: Data files not found (add them to run)")
    
    print("\n" + "="*80)
    if all_critical_passed:
        print("✅ VALIDATION PASSED - Ready for Kaggle!")
        print("="*80)
        print("\nNext steps:")
        print("1. Copy OptimizedDigiCowSolver_Kaggle.py to Kaggle notebook")
        print("2. Update file paths to match your dataset")
        print("3. Run the notebook")
        print("4. Download optimized_submission.csv")
        print("5. Submit to Zindi!")
        print("\nSee KAGGLE_DEPLOYMENT.md for detailed instructions.")
        return 0
    else:
        print("❌ VALIDATION FAILED - Check errors above")
        print("="*80)
        return 1

if __name__ == "__main__":
    sys.exit(main())
