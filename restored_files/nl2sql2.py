"""
NL2SQL V2：加上表结构描述
改进：把表结构注入 Prompt，让 AI 知道数据库长什么样
"""
import os
import pymysql
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# ========== LLM 配置 ==========
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://llm-p5jtn7y87smzjpwo.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)
MODEL = "qwen-plus"

def call_llm(prompt: str) -> str:
    """调用 LLM 并返回纯文本"""
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content

# ========== 数据库配置 ==========
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root123',
    'database': 'school_db',
    'charset': 'utf8mb4'
}

# ========== 新增：表结构描述 ==========
SCHEMA = """
数据库中有两张表：

1. students 表（学生信息）：
   - id: INT, 主键, 学生 ID
   - name: VARCHAR(50), 学生姓名
   - class: VARCHAR(50), 班级
   - age: INT, 年龄

2. scores 表（学生成绩）：
   - id: INT, 主键, 成绩 ID
   - student_id: INT, 外键, 关联 students.id
   - subject: VARCHAR(50), 科目（如：数学、英语）
   - score: DECIMAL(5,2), 分数
   - exam_date: DATE, 考试日期

关系：一个学生有多条成绩记录（一对多）
查询时需要用 student_id 关联两张表。
"""

def generate_sql(question: str) -> str:
    """
    用 LLM 把自然语言转成 SQL
    """
    prompt = f"""
你是一个 SQL 专家。根据用户的问题和数据库表结构，生成对应的 SQL 查询语句。

数据库表结构：
{SCHEMA}

用户问题：{question}

要求：
1. 只返回 SQL 语句，不要其他解释
2. 使用正确的字段名和表名
3. 如果需要关联表，使用 JOIN
"""

    sql = call_llm(prompt).strip()
    sql = sql.replace('```sql', '').replace('```', '').strip()

    return sql

def execute_sql(sql: str):
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            results = cursor.fetchall()
            return results
    finally:
        conn.close()

def nl2sql(question: str):
    print(f"📝 用户问题：{question}")

    sql = generate_sql(question)
    print(f"🔧 生成的 SQL：{sql}")

    try:
        results = execute_sql(sql)
        print(f"✅ 查询结果：{results}")
        return results
    except Exception as e:
        print(f"❌ 执行失败：{e}")
        return None

if __name__ == '__main__':
    nl2sql("所有学生的所有数学考试的数学平均分是多少")