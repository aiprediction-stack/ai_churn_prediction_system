# projects/customer_churn_bank_code/customer_churn_bank_shap_v2.py
# 銀行客戶流失預測 - SHAP 分析 V2 (與 Train V2 和 Config V2 完美對接)

import logging
import warnings
import argparse
import sys
import os 
import re 
import joblib 
import matplotlib.pyplot as plt

# ==========================================
# 1. V2 架構導航 (Nav System)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__)) # projects/code/
project_root = os.path.dirname(os.path.dirname(current_dir)) # WEB_MODEL_MAIN/

if project_root not in sys.path:
    sys.path.append(project_root)

# 加入當前目錄以導入 Train V2 的類別
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from config_v2 import config
    # ★★★ 戰略連結：直接引用 Train V2 的特徵工程邏輯 ★★★
    # 這保證了我們解釋的邏輯跟訓練時一模一樣！
    from projects.customer_churn_bank_code.customer_churn_bank_train_v2 import FeatureEngineer
except ImportError as e:
    # 本地調試用 fallback
    sys.path.append(os.path.join(project_root, 'v2.0x', 'Web_Model_Prediction-main'))
    from config_v2 import config
    try:
        from customer_churn_bank_train_v2 import FeatureEngineer
    except ImportError:
        print("❌ 無法導入 Train V2 或 Config V2，請檢查路徑。")
        sys.exit(1)

# 載入開發環境配置
APP_CONFIG = config['development']

# 設置警告和日誌
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - SHAP_V2 - %(levelname)s - %(message)s')
logger = logging.getLogger('ShapV2')

try:
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    import shap
except ImportError as e:
    logger.error(f"缺少庫: {e}")
    sys.exit(1)

# ==========================================
# 2. SHAP 分析器 (V2 版)
# ==========================================
class ShapAnalyzerV2:
    def __init__(self):
        self.model_path = APP_CONFIG.MODEL_BANK_PATH
        self.meta_dir = APP_CONFIG.MODEL_META_DIR
        self.model = None
        self.feature_cols = None
        self.fe_pipeline_name = None

    def load_artifacts(self) -> bool:
        """從 Config 指定的絕對路徑載入所有裝備"""
        
        # A. 載入模型
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"✅ 模型加載成功: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ 模型加載失敗: {e}")
            return False

        # B. 載入特徵列表 (確保順序一致)
        fc_path = os.path.join(self.meta_dir, 'feature_columns.joblib')
        try:
            self.feature_cols = joblib.load(fc_path)
            logger.info(f"✅ 特徵列表加載成功 ({len(self.feature_cols)} cols)")
        except Exception as e:
            logger.error(f"❌ 特徵列表加載失敗: {e}")
            return False

        # C. 載入 FE 名稱
        fn_path = os.path.join(self.meta_dir, 'fe_pipeline_name.txt')
        try:
            with open(fn_path, 'r') as f:
                self.fe_pipeline_name = f.read().strip()
            logger.info(f"ℹ️ 使用特徵工程: {self.fe_pipeline_name}")
        except Exception:
            logger.warning("⚠️ 無法讀取 FE 名稱，將預設使用 V2")
            self.fe_pipeline_name = 'run_v2_preprocessing'

        return True

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用 Train V2 的邏輯清洗數據"""
        # 1. 取得對應的清洗函數
        fe_func = FeatureEngineer.FE_PIPELINES.get(self.fe_pipeline_name, FeatureEngineer.run_v2_preprocessing)
        
        # 2. 清洗 (注意：這裡是解釋階段，視為 is_train=True 以保留原始分佈，或 False 看需求)
        # 通常為了 SHAP 能看到特徵全貌，我們處理方式與訓練集一致
        df_processed = fe_func(df, is_train=True)
        
        # 3. OHE
        cat_cols = [c for c in df_processed.columns if df_processed[c].dtype.name in ['object', 'str']]
        X_oh = pd.get_dummies(df_processed, columns=cat_cols, dummy_na=False)
        
        # 4. 強制對齊 (Alignment) - 這是 V2 的核心防禦
        missing = set(self.feature_cols) - set(X_oh.columns)
        for c in missing: X_oh[c] = 0.0
        
        # 排序與篩選
        X_final = X_oh[[c for c in self.feature_cols if c in X_oh.columns]]
        return X_final.astype(float)

    def run_shap(self, X_df, n_samples=1000):
        """執行 SHAP 計算並畫圖"""
        if X_df.shape[0] > n_samples:
            X_sample = X_df.sample(n=n_samples, random_state=42)
        else:
            X_sample = X_df

        # --- 黑魔法：修復 XGBoost JSON base_score 問題 ---
        # 這是為了讓 SHAP 能讀懂 XGBoost 模型的必要手段
        final_model = self.model
        temp_json = os.path.join(self.meta_dir, "shap_temp.json")
        
        try:
            booster = self.model.get_booster()
            booster.save_model(temp_json)
            
            with open(temp_json, 'r') as f: content = f.read()
            # Regex 修復 "[0.5]" -> "0.5"
            new_content = re.sub(r'"base_score":\s*"\[(.*?)\]"', r'"base_score": "\1"', content)
            
            with open(temp_json, 'w') as f: f.write(new_content)
            
            clean_booster = xgb.Booster()
            clean_booster.load_model(temp_json)
            final_model = clean_booster
            logger.info("🔧 XGBoost 模型元數據已修復")
            
        except Exception as e:
            logger.warning(f"無法執行 XGBoost 修復，嘗試直接使用原模型: {e}")
        finally:
            if os.path.exists(temp_json): os.remove(temp_json)

        # --- 計算 SHAP ---
        logger.info("🧠 開始計算 SHAP 值...")
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_sample)

        # --- 畫圖並存檔 ---
        # 這裡存到 MODEL_META_DIR，這樣 API 就能讀到了！
        output_path = os.path.join(self.meta_dir, "shap_summary_plot.png")
        
        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"🎨 全局 SHAP 圖已生成: {output_path}")

# ==========================================
# 3. 入口
# ==========================================
def main():
    # 假設訓練數據在同目錄
    data_path = os.path.join(current_dir, "customer_churn_bank_train.csv")
    if not os.path.exists(data_path):
        logger.error(f"找不到數據: {data_path}")
        return

    df = pd.read_csv(data_path)
    
    analyzer = ShapAnalyzerV2()
    if not analyzer.load_artifacts():
        return

    logger.info("正在處理數據...")
    X_final = analyzer.process_data(df)
    
    analyzer.run_shap(X_final, n_samples=2000)
    print("\n🎉 SHAP V2 分析完成！API 現在可以顯示全局解釋圖了。")

if __name__ == "__main__":
    main()