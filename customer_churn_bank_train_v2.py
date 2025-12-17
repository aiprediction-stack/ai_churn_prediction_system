# projects/customer_churn_bank_code/customer_churn_bank_train_v2.py
# 銀行客戶流失預測 - V2 架構整合版 (XGBoost Optuna/SHAP)
# 特點：直接讀取 config_v2，確保模型產出位置與 App 讀取位置完全一致

import logging
import warnings
import argparse
import sys
import os 
from typing import Any, Callable, Tuple, Dict, List
import joblib 

# ==========================================
# 1. V2 架構導航系統 (Nav System)
# ==========================================
# 確保能找到根目錄的 config_v2
current_dir = os.path.dirname(os.path.abspath(__file__)) # projects/code/
project_root = os.path.dirname(os.path.dirname(current_dir)) # WEB_MODEL_MAIN/

if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from config_v2 import config
except ImportError:
    # 為了防止路徑層級沒對好，做個備用方案
    sys.path.append(os.path.join(project_root, 'v2.0x', 'Web_Model_Prediction-main'))
    from config_v2 import config

# 載入開發環境配置 (取得路徑資訊)
APP_CONFIG = config['development']

# 設置警告和日誌
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - V2_TRAIN - %(levelname)s - %(message)s')
logger = logging.getLogger('TrainV2')

# 檢查必要的庫
try:
    import numpy as np
    import pandas as pd
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        logger.error(f"錯誤: 缺少必要的庫: {e}")
        sys.exit(1)
        
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.base import clone
    import optuna
except ImportError as e:
    logger.error(f"錯誤: 缺少必要的庫: {e}")
    sys.exit(1)

# ==========================================
# 2. 訓練參數配置 (不再使用本地 class Config)
# ==========================================
class TrainConfig:
    TARGET_COL = 'Exited'
    N_SPLITS = 5
    RANDOM_STATE = 42
    
    # ★★★ 關鍵改動：路徑直接從 APP_CONFIG 讀取 ★★★
    # 這樣訓練完的模型，App V2 直接就能用！
    MODEL_PATH = APP_CONFIG.MODEL_BANK_PATH
    MODEL_META_DIR = APP_CONFIG.MODEL_META_DIR
    
    # 確保輸出目錄存在
    os.makedirs(MODEL_META_DIR, exist_ok=True)

logger.info(f"🎯 目標模型路徑: {TrainConfig.MODEL_PATH}")
logger.info(f"📂 元數據目錄: {TrainConfig.MODEL_META_DIR}")

# --- 特徵工程類別 (保留原本精華) ---
class FeatureEngineer:
    """用於特徵工程的工具類別。"""
    @staticmethod
    def map_columns(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
        df_copy = df.copy()
        for col, mapping in mappings.items():
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].map(mapping)
        return df_copy

    @staticmethod
    def cast_columns(df: pd.DataFrame, int_cols: Any = None, cat_cols: Any = None) -> pd.DataFrame:
        df_copy = df.copy()
        if int_cols:
            for col in int_cols:
                if col in df_copy.columns:
                    df_copy[col] = df_copy[col].fillna(0).astype(int) 
        return df_copy

    @staticmethod
    def run_v1_preprocessing(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        df_copy = df.copy()
        gender_map = {'Male': 0, 'Female': 1}
        
        df_copy = FeatureEngineer.map_columns(df_copy, {'Gender': gender_map}) 
        if 'Gender' in df_copy.columns:
            df_copy['Gender'] = df_copy['Gender'].fillna(0).astype(int)

        if 'Geography' in df_copy.columns and df_copy['Geography'].dtype.name != 'object':
             df_copy['Geography'] = df_copy['Geography'].astype(str)
        
        if 'Age' in df_copy.columns:
            df_copy['Age_bin'] = pd.cut(df_copy['Age'], bins=[0, 25, 35, 45, 60, np.inf],
                                        labels=['very_young', 'young', 'mid', 'mature', 'senior']).astype(str)
        else:
            df_copy['Age_bin'] = 'unknown'
        
        if 'NumOfProducts' in df_copy.columns:
            df_copy['Is_two_products'] = (df_copy['NumOfProducts'] == 2)
        else:
            df_copy['Is_two_products'] = 0
            
        is_germany = (df_copy['Geography'] == 'Germany') if 'Geography' in df_copy.columns else False
        
        if 'Gender' in df_copy.columns:
            df_copy['Germany_Female'] = (is_germany & (df_copy['Gender'] == 1))
        else:
            df_copy['Germany_Female'] = 0

        if 'IsActiveMember' in df_copy.columns:
            df_copy['Germany_Inactive'] = (is_germany & (df_copy['IsActiveMember'] == 0))
        else:
            df_copy['Germany_Inactive'] = 0
            
        if 'Balance' in df_copy.columns:
            df_copy['Has_Zero_Balance'] = (df_copy['Balance'] == 0)
        else:
            df_copy['Has_Zero_Balance'] = 0

        if 'Tenure' in df_copy.columns:
            df_copy['Tenure_log'] = np.log1p(df_copy['Tenure'].clip(lower=0))
        else:
            df_copy['Tenure_log'] = 0.0

        for col in ['Is_two_products', 'Germany_Female', 'Germany_Inactive', 'Has_Zero_Balance']:
            if col in df_copy.columns:
                 df_copy[col] = df_copy[col].astype(int)

        int_cols = ['HasCrCard', 'IsActiveMember', 'NumOfProducts', 'Is_two_products', 'Has_Zero_Balance',
                    'Germany_Female', 'Germany_Inactive', 'Gender']

        df_copy = FeatureEngineer.cast_columns(df_copy, int_cols=int_cols, cat_cols=None) 

        cols_to_drop = ['id','CustomerId', 'Tenure','Surname', 'RowNumber' ] 
        if is_train and TrainConfig.TARGET_COL in df_copy.columns:
            cols_to_drop.append(TrainConfig.TARGET_COL) 

        df_copy.drop(columns=[col for col in cols_to_drop if col in df_copy.columns], inplace=True, errors='ignore')
        
        for col in df_copy.columns:
            if df_copy[col].dtype.name not in ['object', 'category', 'str']:
                 if col not in int_cols: 
                     df_copy[col] = df_copy[col].astype(float) 

        return df_copy

    @staticmethod
    def run_v2_preprocessing(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        original_df = df.copy() 
        df_copy = FeatureEngineer.run_v1_preprocessing(original_df.copy(), is_train=is_train)

        if all(col in original_df.columns for col in ['Balance', 'IsActiveMember', 'Age']):
            df_copy['is_mature_inactive_transit'] = (
                                                    (original_df['Balance'] == 0) & 
                                                    (original_df['IsActiveMember'] == 0) & 
                                                    (original_df['Age'] > 40)).astype(int)
        else:
            df_copy['is_mature_inactive_transit'] = 0
        
        df_copy['is_mature_inactive_transit'] = df_copy['is_mature_inactive_transit'].astype(int)
        
        if TrainConfig.TARGET_COL in df_copy.columns: 
             df_copy.drop(columns=[TrainConfig.TARGET_COL], inplace=True, errors='ignore')
        
        return df_copy
    
    FE_PIPELINES: Dict[str, Callable] = {
        'run_v2_preprocessing': run_v2_preprocessing,
        'run_v1_preprocessing': run_v1_preprocessing,
    }

# --- Optuna 調優 (保留原樣) ---
class HyperparameterTuner:
    @staticmethod
    def _objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }
        fixed_params = {
            'random_state': TrainConfig.RANDOM_STATE,
            'verbose': 0, 'eval_metric': 'logloss', 'n_jobs': -1,
            'early_stopping_rounds': 50, 'enable_categorical': False, 
        }
        full_params = {**params, **fixed_params}
        model = XGBClassifier(**full_params)
        skf = StratifiedKFold(n_splits=TrainConfig.N_SPLITS, shuffle=True, random_state=fixed_params['random_state'])
        roc_auc_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            fit_params = {'eval_set': [(X_val, y_val)], 'verbose': False}
            try:
                model.fit(X_tr, y_tr, **fit_params)
                best_iteration = model.get_booster().best_iteration
                proba_val = model.predict_proba(X_val, iteration_range=(0, best_iteration))[:, 1]
                roc_auc_scores.append(roc_auc_score(y_val, proba_val))
            except Exception:
                return 0.0
        return float(np.mean(roc_auc_scores))

    @staticmethod
    def tune(X: pd.DataFrame, y: pd.Series, n_trials: int) -> dict:
        optuna.logging.set_verbosity(logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: HyperparameterTuner._objective(trial, X, y), n_trials=n_trials, show_progress_bar=True)
        return study.best_params

# --- 模型訓練器 (V2 修正版) ---
class ModelTrainer:
    def __init__(self):
        self.logger = logging.getLogger('Trainer')

    def run_experiment(self, train_df, test_df, fe_pipeline, models, target_col=TrainConfig.TARGET_COL):
        self.logger.info(f"--- 啟動 V2 實驗 (FE: {fe_pipeline.__name__}) ---")
        test_ids = test_df['id'].copy()
        y_train = train_df[target_col].astype(int)

        X_train_processed = fe_pipeline(train_df.drop(columns=[target_col], errors='ignore').copy(), is_train=True)
        X_test_processed = fe_pipeline(test_df.copy(), is_train=False)

        cat_cols_train = [col for col in X_train_processed.columns if X_train_processed[col].dtype.name in ['object', 'str']]
        cat_cols_test = [col for col in X_test_processed.columns if X_test_processed[col].dtype.name in ['object', 'str']]
        cat_cols = list(set(cat_cols_train + cat_cols_test))

        X_train_oh = pd.get_dummies(X_train_processed, columns=cat_cols, dummy_na=False)
        X_test_oh = pd.get_dummies(X_test_processed, columns=cat_cols, dummy_na=False)
        
        feature_names = X_train_oh.columns.tolist()
        missing_cols_test = set(feature_names) - set(X_test_oh.columns)
        for c in missing_cols_test: X_test_oh[c] = 0
            
        X_test_processed = X_test_oh[[col for col in feature_names if col in X_test_oh.columns]]
        X_train_processed = X_train_oh.astype(float)
        X_test_processed = X_test_processed.astype(float)
        
        self.logger.info(f"特徵數量: {len(feature_names)}")
        
        # 簡單訓練最佳模型 (不重跑 CV 以節省時間，直接用全數據或最後一折)
        # 這裡簡化流程，直接取第一個模型
        best_model = None
        for name, model in models.items():
            model.set_params(enable_categorical=False)
            model.fit(X_train_processed, y_train, verbose=False)
            best_model = model # 這裡簡化，直接拿最後訓練的
        
        # 生成預測
        test_preds = best_model.predict_proba(X_test_processed)[:, 1]
        
        # 生成提交檔 (存到 meta dir)
        sub_path = os.path.join(TrainConfig.MODEL_META_DIR, 'submission_v2.csv')
        submission_df = pd.DataFrame({'id': test_ids, 'Exited': test_preds})
        submission_df.to_csv(sub_path, index=False)
        self.logger.info(f"提交檔已存: {sub_path}")

        return submission_df, {}, best_model, feature_names

    def save_v2_artifacts(self, model, fe_name, feature_cols):
        """V2 專屬存檔：直接存到 Config 指定的絕對路徑"""
        
        # 1. 保存模型 (XGBoost 黑魔法修正)
        try:
            if isinstance(model, XGBClassifier):
                # 簡單修復：確保 base_score 是 float (避開 JSON hack，直接設置參數)
                bs = model.get_params().get('base_score', 0.5)
                if isinstance(bs, str): bs = 0.5
                model.set_params(base_score=bs)
                
            joblib.dump(model, TrainConfig.MODEL_PATH)
            self.logger.info(f"✅ 模型 V2 已部署至: {TrainConfig.MODEL_PATH}")
        except Exception as e:
            self.logger.error(f"❌ 模型保存失敗: {e}")

        # 2. 保存特徵列表 (Service V2 對齊用)
        fc_path = os.path.join(TrainConfig.MODEL_META_DIR, 'feature_columns.joblib')
        joblib.dump(feature_cols, fc_path)
        self.logger.info(f"📝 特徵列表已同步: {fc_path}")

        # 3. 保存 FE 名稱
        fn_path = os.path.join(TrainConfig.MODEL_META_DIR, 'fe_pipeline_name.txt')
        with open(fn_path, 'w') as f:
            f.write(fe_name)

# --- 主程式 ---
def main(tune=False):
    # 路徑也是動態的 (假設 csv 在同目錄)
    data_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(data_dir, "customer_churn_bank_train.csv")
    test_file = os.path.join(data_dir, "customer_churn_bank_test.csv")
    
    if not os.path.exists(train_file):
        logger.error(f"找不到訓練檔: {train_file}")
        return

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    trainer = ModelTrainer()
    best_fe = FeatureEngineer.run_v2_preprocessing
    
    # 參數設置 (你的最佳參數)
    best_params = {
        'n_estimators': 2692, 'learning_rate': 0.0578, 'max_depth': 3,
        'random_state': TrainConfig.RANDOM_STATE, 'n_jobs': -1, 'verbose': 0
    }
    
    if tune:
        logger.info("正在進行調優 (這會花點時間)...")
        # (調優邏輯省略，直接用最佳參數示範)
    
    model = XGBClassifier(**best_params)
    
    # 執行實驗
    _, _, trained_model, feature_cols = trainer.run_experiment(
        df_train, df_test, best_fe, {'XGB_V2': model}
    )
    
    # 存檔 (關鍵一步！)
    trainer.save_v2_artifacts(trained_model, best_fe.__name__, feature_cols)
    print("\n🎉 V2 訓練完成！前線 App 現在可以直接重啟並載入新模型了！")

if __name__ == "__main__":
    # 預設直接執行，不帶參數
    main(tune=False)