# services/business_rules_service.py
import numpy as np
import logging
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger('BusinessRulesService')
logger.setLevel(logging.INFO)

class BusinessRulesService:
    """處理非模型預測相關的業務決策和成本效益分析 (LTV/ENR)。"""

    # --- LTV/ENR 核心參數 ---
    NIM_RATE = 0.02              # 利率
    PRODUCT_PROFIT = 50.0        # 產品利潤
    ACTIVE_CARD_PROFIT = 30.0    # 信用卡利潤
    L_MAX = 10.0                 # 最高存留年限
    USER_RETENTION_COST = 500.0  # 挽留成本
    USER_SUCCESS_RATE = 0.20     # 挽留成功率
    
    def __init__(self):
        logger.info("✅ ROI 業務規則服務初始化成功。")

    def _format_currency(self, amount: float) -> str:
        return f"${amount:,.2f}"

    # ==========================================
    # 1. 單筆計算 (邏輯核心)
    # ==========================================
    def calculate_churn_roi(self, customer_input: Dict[str, Any], proba_churn: float) -> Dict[str, Any]:
        """
        單筆計算 ROI 邏輯。
        注意：這裡包含了所有的商業公式。
        """
        
        # 1. 數據準備
        age = customer_input.get('Age', 40)
        balance = customer_input.get('Balance', 0)
        num_products = customer_input.get('NumOfProducts', 1)
        has_cr_card = customer_input.get('HasCrCard', 0)
        is_active = customer_input.get('IsActiveMember', 0)

        # 確保 proba_churn 避免為 0
        prob_safe = np.maximum(proba_churn, 1e-6)

        # 2. LTV/ENR 公式核心計算
        # 年淨利
        active_card_flag = int((has_cr_card == 1) and (is_active == 1))
        annual_profit = (
            (balance * self.NIM_RATE) +
            (num_products * self.PRODUCT_PROFIT) +
            (active_card_flag * self.ACTIVE_CARD_PROFIT)
        )
        
        # LTV
        expected_lifespan = np.minimum(1 / prob_safe, self.L_MAX)
        ltv = annual_profit * expected_lifespan
        
        # ENR
        enr = (ltv * prob_safe * self.USER_SUCCESS_RATE) - self.USER_RETENTION_COST
        
        # 3. 業務決策
        expected_loss = ltv * prob_safe
        is_worth_spending = bool(enr > 0)
        
        if enr > 1000:
            action = "極高 ROI 潛力，必須實施高階挽留方案"
        elif enr > 0:
            action = "挽留效益為正，建議標準挽留行動"
        elif expected_loss > 10000: 
            action = "高風險，ROI 較低，可考慮成本更低的關懷方案"
        else:
            action = "低風險/低價值，無需主動挽留"

        # 4. 回傳
        return {
            "action_suggested": action,
            "customer_value": self._format_currency(ltv),
            "expected_loss": self._format_currency(expected_loss),
            "net_benefit_if_retained": self._format_currency(enr),
            "is_worth_spending": is_worth_spending
        }

    # ==========================================
    # 2. 批次計算 (呼叫上方的單筆計算)
    # ==========================================
    def calculate_batch_roi(self, input_df: pd.DataFrame, result_df: pd.DataFrame) -> list:
        """
        批次計算 ROI。
        這裡只負責跑迴圈，具體怎麼算，是呼叫上面的 calculate_churn_roi。
        """
        batch_results = []
        
        if 'Exited_Probability' not in result_df.columns:
            logger.error("批次預測結果缺少 Exited_Probability 欄位")
            return []

        input_records = input_df.to_dict('records')
        probabilities = result_df['Exited_Probability'].tolist()
        ids = result_df['id'].tolist() if 'id' in result_df.columns else range(len(input_df))

        for i, customer_data in enumerate(input_records):
            proba = probabilities[i]
            customer_id = ids[i]
            
            # ♻️ 重複利用上方的邏輯
            roi_result = self.calculate_churn_roi(customer_data, proba)
            
            roi_result['id'] = customer_id
            roi_result['probability'] = f"{proba:.4f}"
            
            batch_results.append(roi_result)
            
        return batch_results