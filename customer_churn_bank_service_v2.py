# services/customer_churn_bank_service_v2.py
import pandas as pd
import numpy as np
import logging
import joblib
import shap
import os
import xgboost as xgb
from typing import Dict, Any
from services.model_interface import ModelInterface

logger = logging.getLogger('CustomerChurnBankServiceV2')
logger.setLevel(logging.INFO)

class CustomerChurnBankServiceV2(ModelInterface):
    def __init__(self, model_path: str, model_meta_dir: str):
        self.model = self._load_model(model_path)
        # 定義 V3 模型所需的最終特徵列表 (順序必須一致)
        self.feature_cols = [
            'CreditScore', 'Gender', 'Age', 'Tenure', 'NumOfProducts', 
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
            'Geography_Germany', 'Geography_France', 'Geography_Spain', 
            'Has_Balance', 'Balance_log'
        ]
        self.explainer = None
        self._init_explainer()

    def _load_model(self, path):
        try:
            # 嘗試加載 Joblib (通常是用戶上傳的格式)
            return joblib.load(path)
        except Exception as e:
            logger.warning(f"Joblib 載入失敗，嘗試載入 XGBoost JSON/UBJSON: {e}")
            try:
                # 支援直接載入 XGBoost JSON 格式 (Notebook 中使用的格式)
                model = xgb.XGBClassifier()
                model.load_model(path.replace('.joblib', '.json'))
                return model
            except Exception as e2:
                logger.error(f"模型載入失敗: {e2}")
                return None

    def _init_explainer(self):
        if self.model:
            try:
                # 建立 SHAP Explainer
                self.explainer = shap.TreeExplainer(self.model)
            except Exception as e:
                logger.warning(f"SHAP Explainer 初始化失敗: {e}")

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        V3 特徵工程邏輯 (對應 Notebook 中的 FeatureEngineer.run_v3_preprocessing)
        """
        df_copy = df.copy()
        
        # 1. 性別映射 (Gender Map)
        gender_map = {'Male': 0, 'Female': 1}
        if df_copy['Gender'].dtype == 'object':
             df_copy['Gender'] = df_copy['Gender'].map(gender_map)

        # 2. 國家 One-Hot Encoding (手動處理以確保單筆預測時欄位存在)
        # Notebook 邏輯: Germany, France, Spain
        if 'Geography' in df_copy.columns:
            df_copy['Geography_Germany'] = (df_copy['Geography'] == 'Germany').astype(int)
            df_copy['Geography_France'] = (df_copy['Geography'] == 'France').astype(int)
            df_copy['Geography_Spain'] = (df_copy['Geography'] == 'Spain').astype(int)
        else:
             # 如果輸入資料已經沒有 Geography (例如批次處理時可能發生)，補 0
             for col in ['Geography_Germany', 'Geography_France', 'Geography_Spain']:
                 if col not in df_copy.columns: df_copy[col] = 0

        # 3. 餘額特徵 (V2: Has_Balance, V3: Balance_log)
        if 'Balance' in df_copy.columns:
            # V2: 是否有餘額
            df_copy['Has_Balance'] = (df_copy['Balance'] > 0).astype(int)
            # V3: Log 轉換
            df_copy['Balance_log'] = np.log1p(df_copy['Balance'])
        
        # 4. 移除不需要的欄位 (包含原始的 Geography 和 Balance)
        cols_to_drop = ['id', 'CustomerId', 'Surname', 'Geography', 'Balance', 'Exited']
        df_copy.drop(columns=[c for c in cols_to_drop if c in df_copy.columns], inplace=True, errors='ignore')
        
        # 5. 類型轉換 (確保整數欄位為 int)
        int_cols = ['HasCrCard', 'IsActiveMember', 'Gender', 'Geography_Spain', 'Geography_France', 'Geography_Germany', 'Has_Balance', 'NumOfProducts', 'Tenure']
        for col in int_cols:
            if col in df_copy.columns:
                 df_copy[col] = df_copy[col].fillna(0).astype(int)

        # 6. 特徵對齊 (確保欄位順序與訓練時完全一致)
        # 補齊缺失欄位
        for col in self.feature_cols:
            if col not in df_copy.columns:
                df_copy[col] = 0
                
        # 只保留模型需要的特徵並排序
        return df_copy[self.feature_cols].astype(float)

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.model:
            return {"error": "Model not loaded", "probability": 0.5, "prediction": 0}
        
        X = self.preprocess(df)
        
        try:
            # 預測機率
            prob = self.model.predict_proba(X)[:, 1][0]
        except AttributeError:
            # 兼容沒有 predict_proba 的 Booster 物件
            dmatrix = xgb.DMatrix(X, feature_names=self.feature_cols)
            prob = self.model.predict(dmatrix)[0]

        shap_values_dict = {}
        if self.explainer:
            try:
                # 計算 SHAP 值
                shap_vals = self.explainer.shap_values(X)
                # 處理不同版本的 SHAP 返回格式 (list 或 array)
                vals = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
                
                # 映射特徵名稱與數值
                shap_dict = dict(zip(self.feature_cols, vals))
                # 排序並取前 10 個重要特徵
                shap_values_dict = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
            except Exception as e:
                logger.warning(f"SHAP 計算錯誤: {e}")
                pass

        return {
            "prediction": int(prob >= 0.5),
            "probability": float(prob),
            "local_shap_values": shap_values_dict
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.model: raise RuntimeError("Model not loaded")
        X = self.preprocess(df)
        probs = self.model.predict_proba(X)[:, 1]
        result = df.copy()
        result['Exited_Probability'] = probs
        result['Exited_Prediction'] = (probs >= 0.5).astype(int)
        return result