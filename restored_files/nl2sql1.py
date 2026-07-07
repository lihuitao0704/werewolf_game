"""
NL2SQL V1：最简版本
功能：接收自然语言，生成 SQL，执行并返回结果
"""
import os
from openai import OpenAI
import pymysql
from dotenv import load_dotenv
load_dotenv()

# ========== LLM 配置 ==========
client = OpenAI(
    api_key=os.getenv("LONGCAT_API_KEY"),  # 若没有配置环境变量，请用百炼API Key替换为：api_key="sk-xxx"
    base_url="https://api.longcat.chat/openai",
)
MODEL = "LongCat-2.0"  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models

def call_llm(prompt: str) -> str:
    """调用 LLM 并返回纯文本"""
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content

# ========== 数据库配置 ==========
DB_CONFIG = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '123456',
    'database': 'aistudy',
    'charset': 'utf8mb4'
}

# ========== 核心函数 ==========
def generate_sql(question: str) -> str:
    """
    用 LLM 把自然语言转成 SQL
    """
    prompt = f"""
    你是一个 MYSQL 专家。根据用户的问题，生成对应的 SQL 查询语句。

    用户问题：{question}

    只返回 SQL 语句，不要其他解释。
    """

    sql = call_llm(prompt).strip()

    # 清理：去掉可能的 markdown 标记
    sql = sql.replace('```sql', '').replace('```', '').strip()

    return sql

def execute_sql(sql: str):
    """
    执行 SQL 并返回结果
    """
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            results = cursor.fetchall()
            return results
    except pymysql.err.ProgrammingError as e:
        print(f"❌ SQL 执行出错：{e}")
        print(f"💡 原因：AI 不知道表结构，猜错了字段名")
        return None
    finally:
        conn.close()

def nl2sql(question: str):
    """
    完整流程：自然语言 → SQL → 执行 → 返回结果
    """
    print(f"📝 用户问题：{question}")

    # Step 1: 生成 SQL
    sql = generate_sql(question)
    print(f"🔧 生成的 SQL：{sql}")

    # Step 2: 执行 SQL
    results = execute_sql(sql)
    if results is not None:
        print(f"✅ 查询结果：{results}")

    return results

# ========== 测试 ==========
if __name__ == '__main__':
    nl2sql("student_id的大学英语成绩是多少？")