import pandas as pd
from openai import OpenAI
import time
import requests

# 定义变量
INPUT_FILE = r'output\角色表.csv'  # 输入的CSV文件路径
OUTPUT_FILE = 'translated_chacteristic.csv'  # 输出的CSV文件路径
API_KEY = "你的apikey"  # DeepSeek API密钥
BATCH_SIZE = 100  # 每批处理的数据量
SOURCE_LANG = 'en'  # 源语言
TARGET_LANG = 'zh'  # 目标语言
BASE_URL = "https://api.deepseek.com"  # DeepSeek API的基础URL
MODEL = "deepseek-chat"  # DeepSeek模型名称
DELAY = 0.2  # 每批处理后的延迟时间（秒）
MAX_TOKENS = 8000  # 最大生成的token数
TEMPERATURE = 1.3  # 控制生成文本的多样性
MAX_RETRIES = 3  # 最大重试次数
TIMEOUT = 40  # 增加超时时间

# 初始化DeepSeek客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 批量翻译函数
def translate_batch(texts, source_lang=SOURCE_LANG, target_lang=TARGET_LANG):
    try:
        print("Constructing prompt...")
        prompt = (
"""
你是一名精通日本动漫和全球动漫的专家，熟悉早期和现代动漫、游戏中的角色名称，角色可能来自日本动漫、全球动漫、手游、PC游戏或主机游戏，角色名字可能由日本罗马音或英语构成。你的任务是将角色名称翻译为中文，并**严格**按照指定格式输出。

输出格式：
- 中文译名[别名]-中文系列名-英文系列名。
- 若无别名，则忽略别名部分。
- 每行一条翻译一条文本，**不要添加序号**。
- **如果无法确定角色的系列名称，请只翻译角色名，并在后面跟随英语原名，这是例子：休伯特·冯·维斯特拉-hubert_von_vestra。**
-要翻译每一条tag，让输入行数=输出行数。

输出示例：
阿米娅[兔兔]-明日方舟-arknights
初音未来-初音未来-VOCALOID
休伯特·冯·维斯特拉-hubert_von_vestra

仅输出格式化的内容，不要包含规则之外的任何内容。不要添加任何解释或额外内容。这是接下来要翻译的文本：
"""
            f"{texts}"
        )
        
        print("Calling DeepSeek API...")
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "你是一名专注于 Danbooru 的高级 AI 助手，核心任务是将 Danbooru 的英文标签准确翻译为中文。"},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    stream=False,
                    timeout=TIMEOUT
                )
                print("API Response Received.")
                translated_text = response.choices[0].message.content.strip()
                print("Raw API Response Content:", translated_text)
                # 将翻译结果按逗号分割，并去除空格和空行
                translated_tags = [tag.strip() for tag in translated_text.split(",") if tag.strip()]
                return translated_tags
            except requests.exceptions.Timeout:
                print(f"Attempt {attempt + 1} failed: Request timed out. Retrying...")
                time.sleep(DELAY)  # 延迟后重试
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(DELAY)  # 延迟后重试
        
        print("Max retries reached. Translation failed for this batch.")
        return None
    except Exception as e:
        print(f"Error translating batch: {e}")
        return None

# 主程序
if __name__ == "__main__":
    # 读取CSV文件（没有列名）
    try:
        df = pd.read_csv(INPUT_FILE, header=None)
    except FileNotFoundError:
        print(f"Error: CSV file '{INPUT_FILE}' not found. Please check the file path.")
        exit()

    # 获取所有文本
    texts = df[0].tolist()

    # 分批处理
    translated_tags = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        print(f"Processing batch {i // BATCH_SIZE + 1} of {len(texts) // BATCH_SIZE + 1}...")
        translated_batch = translate_batch(batch)
        if translated_batch:
            translated_tags.extend(translated_batch)
        time.sleep(DELAY)  # 延迟以避免API速率限制

    # 将翻译结果保存到CSV文件
    if translated_tags:
        # 创建一个新的DataFrame来存储翻译结果
        translated_df = pd.DataFrame(translated_tags, columns=["Translated Tag"])
        # 保存为CSV文件，每行一个tag
        translated_df.to_csv(OUTPUT_FILE, index=False, header=False, encoding='utf-8')
        print(f"Translation saved to '{OUTPUT_FILE}'.")
    else:
        print("Translation failed.")