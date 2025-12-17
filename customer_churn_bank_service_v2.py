# services/customer_churn_bank_service_v2.py
import pandas as pd
import numpy as np
import logging
import joblib
import shap
import os
from typing import Dict, Any
from services.model_interface import ModelInterface

logger = logging.getLogger('CustomerChurnBankServiceV2')
logger.setLevel(logging.INFO)

class CustomerChurnBankServiceV2(ModelInterface):
    def __init__(self, model_path: str, model_meta_dir: str):
        self.model = self._load_model(model_path)
        self.feature_cols, _ = self._load_model_artifacts(model_meta_dir)
        self.explainer = None
        if self.model:
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception:
                pass

    def _load_model(self, path):
        try:
            return joblib.load(path)
        except Exception as e:
            logger.error(f"模型載入失敗: {e}")
            return None

    def _load_model_artifacts(self, meta_dir):
        try:
            cols = joblib.load(os.path.join(meta_dir, 'feature_columns.joblib'))
            return cols, "v2_integrated_pipeline"
        except Exception:
            return [], ""

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """V2 特徵工程邏輯"""
        df_copy = df.copy()
        
        # 1. 基礎轉換
        int_cols = ['HasCrCard', 'IsActiveMember', 'NumOfProducts']
        for col in int_cols:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0).astype(int)

        if 'Gender' in df_copy.columns:
            if df_copy['Gender'].dtype in ['int64', 'float64']:
                df_copy['Gender'] = df_copy['Gender'].replace({0: 'Male', 1: 'Female'})
            df_copy['Gender'] = df_copy['Gender'].astype('category')
        
        geo_map = {0: 'France', 1: 'Spain', 2: 'Germany'}
        if 'Geography' in df_copy.columns:
            if df_copy['Geography'].dtype in ['int64', 'float64']:
                df_copy['Geography'] = df_copy['Geography'].replace(geo_map)
            df_copy['Geography'] = df_copy['Geography'].astype('category')

        # 2. 衍生特徵
        if 'Age' in df_copy.columns:
            df_copy['Age_bin'] = pd.cut(df_copy['Age'], bins=[0, 25, 35, 45, 60, np.inf],
                                      labels=['very_young', 'young', 'mid', 'mature', 'senior'],
                                      right=False).astype('category')
        
        if 'NumOfProducts' in df_copy.columns:
            df_copy['Is_two_products'] = (df_copy['NumOfProducts'] == 2).astype(int)
        
        is_germany = (df_copy['Geography'] == 'Germany')
        if 'Gender' in df_copy.columns:
            df_copy['Germany_Female'] = (is_germany & (df_copy['Gender'] == 'Female')).astype(int)
        if 'IsActiveMember' in df_copy.columns:
            df_copy['Germany_Inactive'] = (is_germany & (df_copy['IsActiveMember'] == 0)).astype(int)
            
        if 'Balance' in df_copy.columns:
            df_copy['Has_Zero_Balance'] = (df_copy['Balance'] == 0).astype(int)
            
        if 'Tenure' in df_copy.columns:
            df_copy['Tenure_log'] = np.log1p(df_copy['Tenure'])

        if all(col in df_copy.columns for col in ['Has_Zero_Balance', 'IsActiveMember', 'Age']):
            df_copy['is_mature_inactive_transit'] = (
                (df_copy['Has_Zero_Balance'] == 1) & 
                (df_copy['IsActiveMember'] == 0) & 
                (df_copy['Age'] > 40)
            ).astype(int)

        # 3. OHE 與 對齊
        cat_cols = [c for c in df_copy.columns if df_copy[c].dtype.name in ['object', 'category']]
        df_processed = pd.get_dummies(df_copy, columns=cat_cols)

        if self.feature_cols:
            missing = set(self.feature_cols) - set(df_processed.columns)
            for c in missing: df_processed[c] = 0
            df_processed = df_processed[self.feature_cols]
            
        return df_processed.astype(float)

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.model:
            return {"error": "Model not loaded", "probability": 0.5, "prediction": 0}
        
        X = self.preprocess(df)
        prob = self.model.predict_proba(X)[:, 1][0]
        
        shap_values = {}
        if self.explainer:
            try:
                shap_vals = self.explainer.shap_values(X)
                vals = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
                shap_dict = dict(zip(X.columns, vals))
                shap_values = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:7])
            except Exception:
                pass

        return {
            "prediction": int(prob >= 0.5),
            "probability": float(prob),
            "local_shap_values": shap_values
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.model: raise RuntimeError("Model not loaded")
        X = self.preprocess(df)
        probs = self.model.predict_proba(X)[:, 1]
        result = df.copy()
        result['Exited_Probability'] = probs
        result['Exited_Prediction'] = (probs >= 0.5).astype(int)
        return result