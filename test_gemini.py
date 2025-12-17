from google import genai
import os

# 🔴 請將您的 Key 貼在這裡
MY_API_KEY = "AIzaSyCz5X7qPjdeSOoKpfXkOCZjrvIH1kkEnoA" 

print(f"正在測試 Key: {MY_API_KEY[:10]}...")

try:
    client = genai.Client(api_key=MY_API_KEY)
    
    print("\nAttempt 1: 測試列出所有可用模型...")
    # 列出您的帳號可用的所有模型
    available_models = []
    for m in client.models.list():
        # 過濾出包含 flash 的模型方便查看
        if 'flash' in m.name:
            print(f"✅ 可用模型: {m.name}")
            available_models.append(m.name)
    
    if not available_models:
        print("⚠️ 未找到包含 'flash' 的模型，請檢查您的 Key 權限。")
    
    print("\nAttempt 2: 測試生成內容...")
    # 使用找到的第一個模型，或是預設模型進行測試
    test_model = available_models[0] if available_models else "gemini-1.5-flash-001"
    
    # 注意：SDK 有時回傳的名稱包含 'models/' 前綴，呼叫時可以保留或去掉，通常去掉較保險
    if "/" in test_model:
        test_model = test_model.split("/")[-1]
        
    print(f"正在嘗試使用模型: {test_model}")
    response = client.models.generate_content(
        model=test_model, 
        contents="Hello, Gemini!"
    )
    print(f"🎉 成功！回應內容: {response.text}")

except Exception as e:
    print(f"\n❌ 發生錯誤: {e}")