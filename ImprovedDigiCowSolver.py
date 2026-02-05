"""
DIGICOW - COMPREHENSIVE IMPROVED SOLUTION
Based on Deep Data Analysis

KEY FINDINGS:
1. Cooperative membership is HUGE predictor (28% vs 13% for 7-day adoption)
2. Registration type matters (USSD 50% vs Manual 32% for 120-day)
3. Specific topics have 80-90% adoption rates (Biodeal, CRV, Sistema)
4. Temporal patterns: Month matters (Dec/Jan 75%+, vs May/Jun <10%)
5. Farmers with multiple trainings adopt 2-3x more
6. Year 2025 has 3x higher 7-day adoption than 2024
7. has_topic_trained_on separates 0% vs 16.5% adoption
8. Topic count sweet spot: 10-16 topics = 60-95% adoption

IMPROVEMENTS IMPLEMENTED:
- Rich topic encoding (individual topic flags + categories)
- Temporal features (month seasonality, year trend, day of week)
- Farmer history features (is this repeat training?)
- Better target encoding for geography
- Topic diversity and quality metrics
- County-trainer interactions
- Ensemble of models per horizon
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier, 
                               ExtraTreesClassifier, GradientBoostingClassifier)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_FOLDS = 10
CLIP_MIN = 0.00001
CLIP_MAX = 0.99999

class ImprovedDigiCowSolver:
    def __init__(self):
        self.topic_encoders = {}
        self.geo_encoders = {}
        self.trainer_encoders = {}
        self.farmer_history = {}
        
    def parse_topics(self, topic_str):
        """Parse topic string into list"""
        if pd.isna(topic_str) or topic_str == '[]':
            return []
        try:
            topic_str = topic_str.strip()
            if topic_str.startswith('[') and topic_str.endswith(']'):
                topic_str = topic_str[1:-1]
            topics = [t.strip().strip("'\"") for t in topic_str.split(',') if t.strip()]
            return topics
        except:
            return []
    
    def create_comprehensive_features(self, df, is_train=True, targets=None):
        """Create all features based on data analysis"""
        df = df.copy()
        
        print(f"  Creating features for {len(df)} records...")
        
        # ============ BASIC DEMOGRAPHICS ============
        df['is_female'] = (df['gender'] == 'Female').astype(int)
        df['is_ussd'] = (df['registration'] == 'Ussd').astype(int)
        df['age_above35'] = (df['age'] == 'Above 35').astype(int)
        df['in_coop'] = df['belong_to_cooperative'].fillna(0).astype(int)
        df['has_topic_flag'] = df['has_topic_trained_on'].fillna(0).astype(int)
        
        # Interactions
        df['female_coop'] = df['is_female'] * df['in_coop']
        df['ussd_coop'] = df['is_ussd'] * df['in_coop']
        df['female_ussd'] = df['is_female'] * df['is_ussd']
        df['age_coop'] = df['age_above35'] * df['in_coop']
        
        # ============ TEMPORAL FEATURES ============
        df['date'] = pd.to_datetime(df['training_date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Year is critical (2025 vs 2024)
        df['is_2025'] = (df['year'] == 2025).astype(int)
        
        # Month groupings based on analysis
        df['is_high_adopt_month'] = df['month'].isin([11, 12, 1, 2]).astype(int)  # Nov-Feb: 50%+
        df['is_medium_adopt_month'] = df['month'].isin([3, 4, 9, 10]).astype(int)  # Mar-Apr, Sep-Oct: 20-60%
        df['is_low_adopt_month'] = df['month'].isin([5, 6, 7, 8]).astype(int)  # May-Aug: <10%
        
        # Agricultural seasons (Kenya)
        df['is_long_rains'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_short_rains'] = df['month'].isin([10, 11]).astype(int)
        df['is_dry_season'] = df['month'].isin([1, 2, 6, 7, 8, 9, 12]).astype(int)
        
        # Day of week (Friday and Saturday slightly better)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # ============ GEOGRAPHY WITH TARGET ENCODING ============
        for geo_col in ['county', 'subcounty', 'ward']:
            if geo_col in df.columns:
                df[geo_col] = df[geo_col].fillna('Unknown')
                
                # Frequency encoding
                if is_train:
                    freq = df[geo_col].value_counts(normalize=True)
                    df[f'{geo_col}_freq'] = df[geo_col].map(freq)
                    self.geo_encoders[f'{geo_col}_freq'] = freq.to_dict()
                    
                    # Target encoding with smoothing
                    if targets is not None:
                        for target_name, target_vals in targets.items():
                            # Create temporary dataframe for groupby
                            temp_df = pd.DataFrame({
                                geo_col: df[geo_col],
                                'target': target_vals
                            })
                            
                            target_mean = target_vals.mean()
                            geo_target = temp_df.groupby(geo_col)['target'].agg(['mean', 'count'])
                            
                            # Smoothing (min 50 samples for full weight)
                            alpha = 50
                            geo_target['smoothed'] = (
                                (geo_target['mean'] * geo_target['count'] + target_mean * alpha) /
                                (geo_target['count'] + alpha)
                            )
                            
                            df[f'{geo_col}_{target_name}_enc'] = df[geo_col].map(geo_target['smoothed'])
                            self.geo_encoders[f'{geo_col}_{target_name}_enc'] = geo_target['smoothed'].to_dict()
                else:
                    # Use pre-computed encodings
                    if f'{geo_col}_freq' in self.geo_encoders:
                        df[f'{geo_col}_freq'] = df[geo_col].map(
                            self.geo_encoders[f'{geo_col}_freq']
                        ).fillna(0.001)
                    
                    for target_name in ['target_07', 'target_90', 'target_120']:
                        enc_name = f'{geo_col}_{target_name}_enc'
                        if enc_name in self.geo_encoders:
                            df[enc_name] = df[geo_col].map(
                                self.geo_encoders[enc_name]
                            ).fillna(df[enc_name].mean() if enc_name in df.columns else 0.15)
        
        # ============ TRAINER ENCODING ============
        if 'trainer' in df.columns:
            df['trainer'] = df['trainer'].fillna('Unknown')
            
            if is_train:
                # Frequency
                trainer_freq = df['trainer'].value_counts(normalize=True)
                df['trainer_freq'] = df['trainer'].map(trainer_freq)
                self.trainer_encoders['trainer_freq'] = trainer_freq.to_dict()
                
                # Target encoding
                if targets is not None:
                    for target_name, target_vals in targets.items():
                        # Create temporary dataframe for groupby
                        temp_df = pd.DataFrame({
                            'trainer': df['trainer'],
                            'target': target_vals
                        })
                        
                        target_mean = target_vals.mean()
                        trainer_target = temp_df.groupby('trainer')['target'].agg(['mean', 'count'])
                        
                        alpha = 100  # Trainers have more samples
                        trainer_target['smoothed'] = (
                            (trainer_target['mean'] * trainer_target['count'] + target_mean * alpha) /
                            (trainer_target['count'] + alpha)
                        )
                        
                        df[f'trainer_{target_name}_enc'] = df['trainer'].map(trainer_target['smoothed'])
                        self.trainer_encoders[f'trainer_{target_name}_enc'] = trainer_target['smoothed'].to_dict()
            else:
                if 'trainer_freq' in self.trainer_encoders:
                    df['trainer_freq'] = df['trainer'].map(
                        self.trainer_encoders['trainer_freq']
                    ).fillna(0.001)
                
                for target_name in ['target_07', 'target_90', 'target_120']:
                    enc_name = f'trainer_{target_name}_enc'
                    if enc_name in self.trainer_encoders:
                        df[enc_name] = df['trainer'].map(
                            self.trainer_encoders[enc_name]
                        ).fillna(df[enc_name].mean() if enc_name in df.columns else 0.15)
        
        # ============ FARMER HISTORY FEATURES ============
        if 'farmer_id' in df.columns:
            # Track farmer training history
            if is_train:
                df_sorted = df.sort_values('date')
                farmer_counts = {}
                training_numbers = []
                
                for idx, row in df_sorted.iterrows():
                    fid = row['farmer_id']
                    if fid not in farmer_counts:
                        farmer_counts[fid] = 0
                    training_numbers.append(farmer_counts[fid])
                    farmer_counts[fid] += 1
                
                df_sorted['training_number'] = training_numbers
                df = df.merge(df_sorted[['ID', 'training_number']], on='ID', how='left')
                self.farmer_history = farmer_counts
            else:
                # For test, we don't know exact history, use proxy
                df['training_number'] = 0  # Conservative assumption
        
        df['is_repeat_training'] = (df.get('training_number', 0) > 0).astype(int)
        df['training_number_log'] = np.log1p(df.get('training_number', 0))
        
        # ============ COMPREHENSIVE TOPIC FEATURES ============
        df = self.create_rich_topic_features(df, is_train, targets)
        
        # ============ INTERACTION FEATURES ============
        # Geography × Time
        if 'county_freq' in df.columns:
            df['county_month'] = df['county_freq'] * df['month']
            df['county_high_month'] = df['county_freq'] * df['is_high_adopt_month']
        
        # Trainer × Time
        if 'trainer_freq' in df.columns:
            df['trainer_month'] = df['trainer_freq'] * df['month']
            df['trainer_year'] = df['trainer_freq'] * df['is_2025']
        
        # Demographics × Topics
        if 'num_topics' in df.columns:
            df['female_topics'] = df['is_female'] * df['num_topics']
            df['coop_topics'] = df['in_coop'] * df['num_topics']
            df['ussd_topics'] = df['is_ussd'] * df['num_topics']
        
        # Cooperative × Geography
        if 'county_freq' in df.columns:
            df['coop_county'] = df['in_coop'] * df['county_freq']
        
        print(f"  Created {len([c for c in df.columns if c not in ['ID', 'farmer_id', 'date', 'topics', 'gender', 'age', 'registration', 'county', 'subcounty', 'ward', 'trainer', 'group_name', 'training_date']])} features")
        
        return df
    
    def create_rich_topic_features(self, df, is_train=True, targets=None):
        """Create comprehensive topic features"""
        df = df.copy()
        
        # Parse topics
        df['topic_list'] = df['topics'].apply(self.parse_topics)
        df['num_topics'] = df['topic_list'].apply(len)
        
        # Topic count bins (based on analysis)
        df['has_1_topic'] = (df['num_topics'] == 1).astype(int)
        df['has_2_5_topics'] = ((df['num_topics'] >= 2) & (df['num_topics'] <= 5)).astype(int)
        df['has_6_9_topics'] = ((df['num_topics'] >= 6) & (df['num_topics'] <= 9)).astype(int)
        df['has_10plus_topics'] = (df['num_topics'] >= 10).astype(int)
        
        # Topic diversity (log scale)
        df['num_topics_log'] = np.log1p(df['num_topics'])
        df['num_topics_sqrt'] = np.sqrt(df['num_topics'])
        
        # High-impact topics (from analysis: 80%+ adoption rate)
        high_impact_topics = [
            'Poultry Health With Biodeal',
            'Herd Health With Biodeal',
            'Pest And Diseases At Harvesting Stage',
            'Livestock Management Practices',
            'How To Manage Pest And Diseases At Harvesting Stage',
            'Successfull Breeding With Crv',
            'Dairy Nutrition With Tyari',
            'Why You Should Vaccinate Your Animals',
            'Weed Management In Crop',
            'Poultry Health Management',
            'Deworming And Record Keeping In Animal Health',
            'Diseases In Dairy Farming',
            'Herd. Health. Management',
            'Dairy Health Management',
        ]
        
        # Medium-impact topics (40-60% adoption)
        medium_impact_topics = [
            'Poultry Feeding With Tyari',
            'Importance Of Mineral Supplementation',
            'Asili Fertilizer (Organic)',
            'Importance Of Choosing The Right Seed Variety',
            'Aflatoxin In Dairy Farming',
            'Antimicrobial Resistance',
            'Milking Hygiene',
        ]
        
        # Low/zero impact topics (0-10% adoption)
        low_impact_topics = [
            'Herd Health. Management',
            'Microp+ Topdressing',
            'Herd Health',
            'Poultry Mngt Practices',
            'Disadvantages Of Natural Mating',
            'Biodeal Poultry',
            'Poultry Management Practices',
            'Reasons Why Ai Fails And Solutions',
            'Herd Management',
            'Poultry Health Mngt',
        ]
        
        # Create flags for topic categories
        df['has_high_impact_topic'] = df['topic_list'].apply(
            lambda topics: any(t in high_impact_topics for t in topics)
        ).astype(int)
        
        df['has_medium_impact_topic'] = df['topic_list'].apply(
            lambda topics: any(t in medium_impact_topics for t in topics)
        ).astype(int)
        
        df['has_low_impact_topic'] = df['topic_list'].apply(
            lambda topics: any(t in low_impact_topics for t in topics)
        ).astype(int)
        
        # Count high-impact topics
        df['num_high_impact'] = df['topic_list'].apply(
            lambda topics: sum(1 for t in topics if t in high_impact_topics)
        )
        
        # Topic quality score
        df['topic_quality'] = (
            df['num_high_impact'] * 3 +
            df['has_medium_impact_topic'] * 1 -
            df['has_low_impact_topic'] * 2
        )
        
        # Individual high-value topic flags (top 10 by adoption)
        top_individual_topics = [
            'Poultry Health With Biodeal',
            'Herd Health With Biodeal',
            'Successfull Breeding With Crv',
            'Why You Should Vaccinate Your Animals',
            'Dairy Nutrition With Tyari',
        ]
        
        for topic in top_individual_topics:
            safe_name = topic.replace(' ', '_').replace('.', '').replace('(', '').replace(')', '').lower()[:30]
            df[f'has_{safe_name}'] = df['topic_list'].apply(
                lambda topics: topic in topics
            ).astype(int)
        
        # Topic theme categories
        df['has_poultry_topic'] = df['topic_list'].apply(
            lambda topics: any('poultry' in t.lower() or 'chicken' in t.lower() for t in topics)
        ).astype(int)
        
        df['has_dairy_topic'] = df['topic_list'].apply(
            lambda topics: any('dairy' in t.lower() or 'milk' in t.lower() or 'cow' in t.lower() for t in topics)
        ).astype(int)
        
        df['has_health_topic'] = df['topic_list'].apply(
            lambda topics: any('health' in t.lower() or 'disease' in t.lower() or 'vaccine' in t.lower() for t in topics)
        ).astype(int)
        
        df['has_feeding_topic'] = df['topic_list'].apply(
            lambda topics: any('feed' in t.lower() or 'nutrition' in t.lower() or 'tyari' in t.lower() for t in topics)
        ).astype(int)
        
        df['has_crop_topic'] = df['topic_list'].apply(
            lambda topics: any('crop' in t.lower() or 'seed' in t.lower() or 'fertilizer' in t.lower() or 'pest' in t.lower() for t in topics)
        ).astype(int)
        
        df['has_breeding_topic'] = df['topic_list'].apply(
            lambda topics: any('breed' in t.lower() or 'ai' in t.lower() or 'ndume' in t.lower() or 'crv' in t.lower() for t in topics)
        ).astype(int)
        
        # Topic mix
        df['num_topic_themes'] = (
            df['has_poultry_topic'] + df['has_dairy_topic'] + 
            df['has_crop_topic'] + df['has_breeding_topic']
        )
        
        df['is_focused_training'] = (df['num_topic_themes'] == 1).astype(int)
        df['is_diverse_training'] = (df['num_topic_themes'] >= 3).astype(int)
        
        return df
    
    def calculate_competition_score(self, y_true, y_pred):
        """Calculate exact competition metric"""
        auc = roc_auc_score(y_true, y_pred)
        ll = log_loss(y_true, y_pred)
        return 0.25 * auc + 0.75 * (1 - ll)
    
    def train_ensemble_model(self, X_train, y_train, X_val, y_val, horizon):
        """Train ensemble of sklearn models for each horizon"""
        
        # Different hyperparameters per horizon
        if horizon == '07':  # Rare events - need class balancing
            hgb_params = {
                'max_iter': 1000,
                'learning_rate': 0.01,
                'max_depth': 7,
                'min_samples_leaf': 20,
                'l2_regularization': 2.0,
                'class_weight': 'balanced',
                'random_state': RANDOM_STATE
            }
            
            rf_params = {
                'n_estimators': 300,
                'max_depth': 12,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
                'random_state': RANDOM_STATE,
                'n_jobs': -1
            }
            
            et_params = {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 8,
                'min_samples_leaf': 4,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
                'random_state': RANDOM_STATE,
                'n_jobs': -1
            }
            
        elif horizon == '90':  # Medium frequency
            hgb_params = {
                'max_iter': 800,
                'learning_rate': 0.02,
                'max_depth': 8,
                'min_samples_leaf': 15,
                'l2_regularization': 1.0,
                'class_weight': 'balanced',
                'random_state': RANDOM_STATE
            }
            
            rf_params = {
                'n_estimators': 250,
                'max_depth': 14,
                'min_samples_split': 8,
                'min_samples_leaf': 4,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
                'random_state': RANDOM_STATE,
                'n_jobs': -1
            }
            
            et_params = {
                'n_estimators': 200,
                'max_depth': 16,
                'min_samples_split': 6,
                'min_samples_leaf': 3,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
                'random_state': RANDOM_STATE,
                'n_jobs': -1
            }
            
        else:  # 120 - more common
            hgb_params = {
                'max_iter': 600,
                'learning_rate': 0.03,
                'max_depth': 9,
                'min_samples_leaf': 10,
                'l2_regularization': 0.5,
                'random_state': RANDOM_STATE
            }
            
            rf_params = {
                'n_estimators': 200,
                'max_depth': 16,
                'min_samples_split': 6,
                'min_samples_leaf': 3,
                'max_features': 'sqrt',
                'random_state': RANDOM_STATE,
                'n_jobs': -1
            }
            
            et_params = {
                'n_estimators': 150,
                'max_depth': 18,
                'min_samples_split': 4,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'random_state': RANDOM_STATE,
                'n_jobs': -1
            }
        
        # Train HGB
        hgb = HistGradientBoostingClassifier(**hgb_params, verbose=0)
        hgb.fit(X_train, y_train)
        
        # Train RF
        rf = RandomForestClassifier(**rf_params, verbose=0)
        rf.fit(X_train, y_train)
        
        # Train ExtraTrees
        et = ExtraTreesClassifier(**et_params, verbose=0)
        et.fit(X_train, y_train)
        
        # Calibrate HGB
        hgb_cal = CalibratedClassifierCV(hgb, cv='prefit', method='isotonic')
        hgb_cal.fit(X_val, y_val)
        
        return {'hgb': hgb_cal, 'rf': rf, 'et': et}
    
    def enforce_monotonic_constraints(self, preds_07, preds_90, preds_120):
        """Enforce P(7) <= P(90) <= P(120)"""
        preds_07 = np.clip(preds_07, CLIP_MIN, CLIP_MAX)
        preds_90 = np.clip(preds_90, CLIP_MIN, CLIP_MAX)
        preds_120 = np.clip(preds_120, CLIP_MIN, CLIP_MAX)
        
        # Enforce monotonicity
        preds_90 = np.maximum(preds_90, preds_07)
        preds_120 = np.maximum(preds_120, preds_90)
        
        # Smooth transitions
        preds_90 = 0.65 * preds_90 + 0.35 * preds_07
        preds_120 = 0.65 * preds_120 + 0.35 * preds_90
        
        return preds_07, preds_90, preds_120
    
    def run_modeling(self, train_df, test_df):
        """Main modeling pipeline"""
        
        print("="*80)
        print("COMPREHENSIVE DIGICOW SOLUTION")
        print("="*80)
        
        # Create target encodings
        targets_for_encoding = {
            'target_07': train_df['adopted_within_07_days'],
            'target_90': train_df['adopted_within_90_days'],
            'target_120': train_df['adopted_within_120_days']
        }
        
        # Create features
        print("\n🔧 Creating comprehensive features...")
        train_features = self.create_comprehensive_features(
            train_df, is_train=True, targets=targets_for_encoding
        )
        test_features = self.create_comprehensive_features(
            test_df, is_train=False
        )
        
        # Define features to use
        exclude_cols = [
            'ID', 'farmer_id', 'training_date', 'topics', 'date',
            'gender', 'age', 'registration', 'county', 'subcounty', 'ward', 
            'trainer', 'group_name', 'topic_list',
            'adopted_within_07_days', 'adopted_within_90_days', 'adopted_within_120_days'
        ]
        
        feature_cols = [col for col in train_features.columns 
                       if col not in exclude_cols and col in test_features.columns]
        
        print(f"Using {len(feature_cols)} features")
        
        # Train for each horizon
        horizons = ['07', '90', '120']
        target_cols = {
            '07': 'adopted_within_07_days',
            '90': 'adopted_within_90_days',
            '120': 'adopted_within_120_days'
        }
        
        predictions = {h: np.zeros(len(test_df)) for h in horizons}
        oof_scores = {h: [] for h in horizons}
        
        for horizon in horizons:
            print(f"\n🎯 Modeling {horizon}-day adoption...")
            target_col = target_cols[horizon]
            y = train_df[target_col].values
            
            # Get features
            X = train_features[feature_cols].values
            X_test = test_features[feature_cols].values
            
            # Stratified K-Fold
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            fold_scores = []
            test_preds_hgb = []
            test_preds_rf = []
            test_preds_et = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                print(f"  Fold {fold+1}/{N_FOLDS}: ", end="", flush=True)
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train ensemble
                models = self.train_ensemble_model(
                    X_train, y_train, X_val, y_val, horizon
                )
                
                # Validate
                y_pred_hgb = models['hgb'].predict_proba(X_val)[:, 1]
                y_pred_rf = models['rf'].predict_proba(X_val)[:, 1]
                y_pred_et = models['et'].predict_proba(X_val)[:, 1]
                
                # Ensemble predictions (weighted)
                y_pred_val = 0.5 * y_pred_hgb + 0.3 * y_pred_rf + 0.2 * y_pred_et
                y_pred_val = np.clip(y_pred_val, CLIP_MIN, CLIP_MAX)
                
                fold_score = self.calculate_competition_score(y_val, y_pred_val)
                fold_scores.append(fold_score)
                
                print(f"Score: {fold_score:.6f}")
                
                # Test predictions
                y_test_hgb = models['hgb'].predict_proba(X_test)[:, 1]
                y_test_rf = models['rf'].predict_proba(X_test)[:, 1]
                y_test_et = models['et'].predict_proba(X_test)[:, 1]
                
                test_preds_hgb.append(y_test_hgb)
                test_preds_rf.append(y_test_rf)
                test_preds_et.append(y_test_et)
            
            # Average test predictions
            avg_hgb = np.mean(test_preds_hgb, axis=0)
            avg_rf = np.mean(test_preds_rf, axis=0)
            avg_et = np.mean(test_preds_et, axis=0)
            predictions[horizon] = 0.5 * avg_hgb + 0.3 * avg_rf + 0.2 * avg_et
            
            oof_scores[horizon] = np.mean(fold_scores)
            
            print(f"  ✅ {horizon}-day CV: {oof_scores[horizon]:.6f} ± {np.std(fold_scores):.6f}")
        
        # Enforce monotonic constraints
        print("\n🔗 Enforcing monotonic constraints...")
        predictions['07'], predictions['90'], predictions['120'] = self.enforce_monotonic_constraints(
            predictions['07'], predictions['90'], predictions['120']
        )
        
        return predictions, oof_scores, feature_cols

def main():
    """Main execution"""
    
    print("\n" + "="*80)
    print("DIGICOW COMPREHENSIVE IMPROVED SOLUTION")
    print("="*80)
    
    # Load data
    print("\n📊 Loading data...")
    train_df = pd.read_csv('/kaggle/input/digicow-new/Train (3).csv')
    test_df = pd.read_csv('/kaggle/input/digicow-new/Test (3).csv')
    
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Initialize solver
    solver = ImprovedDigiCowSolver()
    
    # Run modeling
    predictions, oof_scores, features = solver.run_modeling(train_df, test_df)
    
    # Create submission
    print("\n📝 Creating submission...")
    submission = pd.DataFrame({'ID': test_df['ID']})
    
    for horizon in ['07', '90', '120']:
        pred = predictions[horizon]
        submission[f'Target_{horizon}_AUC'] = pred
        submission[f'Target_{horizon}_LogLoss'] = pred
    
    # Final clip
    for col in submission.columns:
        if 'Target' in col:
            submission[col] = submission[col].clip(CLIP_MIN, CLIP_MAX)
    
    # Save
    filename = 'improved_solution_submission.csv'
    submission.to_csv(filename, index=False)
    print(f"✅ Submission saved: {filename}")
    
    # Calculate final score
    final_score = np.mean(list(oof_scores.values()))
    print(f"\n📈 FINAL ESTIMATED SCORE: {final_score:.6f}")
    
    print("\n" + "="*80)
    print("SCORE BREAKDOWN:")
    print("="*80)
    for horizon in ['07', '90', '120']:
        print(f"{horizon}-day: {oof_scores[horizon]:.6f}")
    
    previous_score = 0.716026163
    if final_score > previous_score:
        improvement = final_score - previous_score
        print(f"\n✅ IMPROVEMENT: +{improvement:.6f} ({improvement/previous_score*100:.2f}%)")
    else:
        print(f"\n⚠️  Score change: {final_score - previous_score:.6f}")
    
    print("\n" + "="*80)
    print("KEY IMPROVEMENTS:")
    print("="*80)
    print("1. ✅ Rich topic encoding (individual topics + categories)")
    print("2. ✅ Target encoding for geography and trainers")
    print("3. ✅ Temporal patterns (month, year, seasonality)")
    print("4. ✅ Farmer history tracking (repeat trainings)")
    print("5. ✅ Ensemble models (HGB + RF + ExtraTrees)")
    print("6. ✅ Comprehensive interactions")
    print("7. ✅ 10-fold CV for robustness")
    print("="*80)
    
    return submission, final_score

if __name__ == "__main__":
    submission, score = main()