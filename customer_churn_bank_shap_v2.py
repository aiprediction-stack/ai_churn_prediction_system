# projects/customer_churn_bank_code/customer_churn_bank_shap_v2.py
# 銀行客戶流失預測 - SHAP 分析 V2 (適配 V3 模型)

import logging
import warnings
import sys
import os 
import joblib 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# 導航設定
current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path: sys.path.append(project_root)

from config_v2 import config

# 載入配置
APP_CONFIG = config['development']

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - SHAP_V2 - %(levelname)s - %(message)s')
logger = logging.getLogger('ShapV2')

class ShapAnalyzerV3:
    def __init__(self):
        self.model_path = APP_CONFIG.MODEL_BANK_PATH
        self.meta_dir = APP_CONFIG.MODEL_META_DIR
        self.model = None
        # V3 特徵列表
        self.feature_cols = [
            'CreditScore', 'Gender', 'Age', 'Tenure', 'NumOfProducts', 
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
            'Geography_Germany', 'Geography_France', 'Geography_Spain', 
            'Has_Balance', 'Balance_log'
        ]

    def load_model(self) -> bool:
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"✅ 模型加載成功: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"❌ 模型加載失敗: {e}")
            return False

    def preprocess_v3(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        執行 V3 預處理邏輯 (與 Service 保持一致)
        """
        df_copy = df.copy()
        
        # 1. Gender Map
        gender_map = {'Male': 0, 'Female': 1}
        if df_copy['Gender'].dtype == 'object':
             df_copy['Gender'] = df_copy['Gender'].map(gender_map)

        # 2. Geography One-Hot
        if 'Geography' in df_copy.columns:
            df_copy['Geography_Germany'] = (df_copy['Geography'] == 'Germany').astype(int)
            df_copy['Geography_France'] = (df_copy['Geography'] == 'France').astype(int)
            df_copy['Geography_Spain'] = (df_copy['Geography'] == 'Spain').astype(int)

        # 3. Balance Features (V3)
        if 'Balance' in df_copy.columns:
            df_copy['Has_Balance'] = (df_copy['Balance'] > 0).astype(int)
            df_copy['Balance_log'] = np.log1p(df_copy['Balance'])
        
        # 4. Fill missing cols with 0
        for col in self.feature_cols:
            if col not in df_copy.columns:
                df_copy[col] = 0

        # 5. Return aligned features
        return df_copy[self.feature_cols].astype(float)

    def run_shap(self, df_raw, n_samples=1000):
        """執行 SHAP 計算並畫圖"""
        logger.info("🔧 執行 V3 預處理...")
        X_df = self.preprocess_v3(df_raw)
        
        if X_df.shape[0] > n_samples:
            X_sample = X_df.sample(n=n_samples, random_state=42)
        else:
            X_sample = X_df

        logger.info("🧠 開始計算 SHAP 值...")
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)

        # 畫圖並存檔
        output_path = os.path.join(self.meta_dir, "shap_summary_plot.png")
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"🎨 全局 SHAP 圖已生成: {output_path}")

def main():
    # 讀取訓練數據 (假設與腳本在同一目錄)
    data_path = os.path.join(current_dir, "customer_churn_bank_train.csv")
    if not os.path.exists(data_path):
        # 嘗試從上層目錄尋找
        data_path = os.path.join(project_root, "customer_churn_bank_train.csv")
    
    if not os.path.exists(data_path):
        logger.error(f"找不到數據文件: {data_path}")
        return

    df = pd.read_csv(data_path)
    
    analyzer = ShapAnalyzerV3()
    if analyzer.load_model():
        analyzer.run_shap(df, n_samples=2000)
        print("\n🎉 V3 SHAP 全局分析圖已更新！")

if __name__ == "__main__":
    main()