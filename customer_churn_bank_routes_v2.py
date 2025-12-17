# routes/customer_churn_bank_routes_v2.py

import matplotlib
# 設定 matplotlib 後端，必須在 pyplot 引入前設定，避免 GUI 錯誤
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import logging
import base64
import io
import sys
import os
from flask import Blueprint, jsonify, request
from utils.api_response import ApiResponse 

# V2 引入
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.append(project_root)

from config_v2 import config
from services.customer_churn_bank_service_v2 import CustomerChurnBankServiceV2
from services.gemini_service_v2 import GeminiServiceV2
from services.business_rules_service import BusinessRulesService

matplotlib.use('Agg')
plt.rcParams['axes.unicode_minus'] = False 
logger = logging.getLogger('CustomerChurnBankRoute')

# 載入配置
env_config = os.getenv('FLASK_CONFIG', 'default')
app_config = config[env_config]

# 初始化服務
SERVICE = CustomerChurnBankServiceV2(app_config.MODEL_BANK_PATH, app_config.MODEL_META_DIR)
GEMINI = GeminiServiceV2(app_config.GEMINI_API_KEY)
BUSINESS_RULES = BusinessRulesService() 


# 載入全局 SHAP
GLOBAL_SHAP_BASE64 = ""
global_path = os.path.join(app_config.MODEL_META_DIR, "shap_summary_plot.png")
if os.path.exists(global_path):
    with open(global_path, "rb") as f: GLOBAL_SHAP_BASE64 = base64.b64encode(f.read()).decode('utf-8')

customer_churn_bank_blueprint = Blueprint('customer_churn_bank_blueprint', __name__)

def generate_local_shap_chart(shap_data: dict, title: str) -> str:
    if not shap_data: return ""
    try:
        sorted_data = dict(sorted(shap_data.items(), key=lambda item: abs(item[1]), reverse=True))
        features, importances = list(sorted_data.keys()), list(sorted_data.values())
        colors = ['#EF5350' if imp > 0 else '#66BB6A' for imp in importances]
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, len(features) * 0.7 + 1))
        ax.barh(features, importances, color=colors)
        ax.set_title(title); ax.invert_yaxis()
        buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception: return ""

@customer_churn_bank_blueprint.route('/predict', methods=['POST'])
def predict_churn():
    try:
        data = request.get_json()
        if not data: return ApiResponse.error("無效的 JSON")
        
        # --- 新增：動態檢查 API Key ---
        client_api_key = request.headers.get('X-Gemini-API-Key')
        if client_api_key:
            # 如果前端有傳 Key，就暫時建立一個新的 Gemini Service 實例
            gemini_service = GeminiServiceV2(client_api_key)
        else:
            # 否則使用預設的全域實例
            gemini_service = GEMINI
        
        # 呼叫 Service V2
        input_df = pd.DataFrame([data])
        result = SERVICE.predict(input_df)
        proba = result.get('probability', 0.5)
        
        # 2. [V2 核心] 呼叫 ROI 服務
        roi_data = BUSINESS_RULES.calculate_churn_roi(data, proba)

        # 呼叫 Gemini V2 (使用剛才決定的 gemini_service)
        shap_values = result.get('local_shap_values', {})
        shap_text = "\n".join([f"{k}: {v:.4f}" for k,v in shap_values.items()])
        
        # 改用 gemini_service 呼叫
        ai_exp = gemini_service.generate_churn_explanation(data, result, shap_text)
        
        # 繪圖
        charts = []
        local_chart = generate_local_shap_chart(shap_values, f"Churn Prob: {proba:.4f}")
        if local_chart: charts.append({"type": "image/png", "base64_data": local_chart, "title": "局部特徵"})
        if GLOBAL_SHAP_BASE64: charts.append({"type": "image/png", "base64_data": GLOBAL_SHAP_BASE64, "title": "全局特徵"})

        payload = {
            "prediction": float(proba),
            "readable_features": data,
            "explanation_prompt": ai_exp,
            "roi_analysis": roi_data, 
            "charts": charts
        }
        return ApiResponse.success(payload)
    except Exception as e:
        return ApiResponse.error(str(e))

@customer_churn_bank_blueprint.route('/predict_batch', methods=['POST'])
def predict_batch():
    if 'file' not in request.files: return ApiResponse.error("未上傳檔案")
    try:
        # 1. 讀取 CSV
        input_df = pd.read_csv(request.files['file'], keep_default_na=True, na_values=['', 'NA'])
        
        # 2. 呼叫模型預測 (取得機率)
        result_df = SERVICE.predict_batch(input_df)
        
        # 補上 ID (如果沒有的話)
        if 'id' not in result_df.columns: result_df['id'] = result_df.index
        
        # 🔥🔥🔥 [修改核心]：原本只回傳機率，現在加入 ROI 批次計算
        # 呼叫我們剛在 Service 裡加上的 calculate_batch_roi 方法
        roi_output_list = BUSINESS_RULES.calculate_batch_roi(input_df, result_df)
        
        # 🔥🔥🔥 [修改回傳]：回傳完整的 ROI 列表
        return ApiResponse.success(roi_output_list, message=f"成功處理 {len(roi_output_list)} 筆並完成 ROI 分析")
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}") # 建議加個 Log
        return ApiResponse.error(str(e))