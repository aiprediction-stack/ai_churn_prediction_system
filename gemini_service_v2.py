# services/gemini_service_v2.py
import logging
from google import genai
from google.genai.errors import APIError

logger = logging.getLogger('GeminiServiceV2')
logger.setLevel(logging.INFO)

class GeminiServiceV2:
    # 🔴 修改這裡：將預設值改為 "gemini-2.0-flash" (根據您測試結果可用的模型)
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        if not api_key:
            logger.warning("⚠️ Gemini API Key 未設定")
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=api_key)
                self.model_name = model_name
                logger.info(f"Gemini V2 Ready (Model: {model_name})")
            except Exception as e:
                logger.error(f"Gemini Init Failed: {e}")
                self.client = None

    def generate_churn_explanation(self, input_features: dict, prediction_result: dict, feature_importance: str) -> str:
        if not self.client:
            return "AI 服務暫時不可用，但根據模型預測，該客戶風險狀況如上所示。"

        prob = float(prediction_result.get('probability', 0.0))
        is_churn = int(prediction_result.get('prediction', 0))
        formatted_features = "\n".join([f"- {k}: {v}" for k, v in input_features.items()])

        tone = "該客戶為**極高流失風險**，請提供強力挽留方案。" if prob > 0.7 else \
               "該客戶有**潛在流失風險**，建議預防性關懷。" if prob > 0.3 else \
               "該客戶**狀況穩定**，尋求交叉銷售機會。"

        prompt = f"""
        你是一位銀行風險分析師。請根據數據生成簡報。
        【分析目標】{tone}
        【客戶數據】{formatted_features}
        【模型診斷】流失機率: {prob:.1%}, 預測: {'⚠️ 流失' if is_churn else '✅ 留存'}
        【關鍵因子】{feature_importance}
        
        請提供：1.風險洞察 2.關鍵因素解讀 3.具體行動建議(ROI導向)。150字內。
        """

        try:
            # 嘗試呼叫 AI
            response = self.client.models.generate_content(model=self.model_name, contents=prompt)
            if not response.text:
                return "AI 回傳內容為空。"
            return response.text.strip()
        except Exception as e:
            # 🔥 重要：將真正的錯誤印在終端機，方便除錯
            print(f"❌ Gemini Error Details: {str(e)}")
            # 回傳包含錯誤部分的訊息，讓前端也能稍微看到
            return f"AI 分析失敗: {str(e)[:50]}..."