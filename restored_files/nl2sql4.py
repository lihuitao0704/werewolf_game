"""
NL2SQL V4：支持完整的增删改查操作
改进：支持 SELECT / INSERT / UPDATE / DELETE 所有 SQL 操作
"""
import os
import json
import pymysql
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ========== LLM 配置 ==========
client = OpenAI(
    api_key=os.getenv("LONGCAT_API_KEY"),  # 若没有配置环境变量，请用百炼API Key替换为：api_key="sk-xxx"
    base_url="https://api.longcat.chat/openai",
)
MODEL = "LongCat-2.0"


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

SCHEMA = """
数据库中有两张表：

1. scores 表（学生成绩表）：
   - id: INT, 主键, 成绩ID
   - student_id: INT,  外键, 关联 students.id
   - subject: VARCHAR(50), 学科
   - score: INT, 年龄

2. students 表（学生信息）：
   - id: INT, 主键, 学生ID
   - name: VARCHAR(50), 学生姓名
   - gender: enum('男','女'), 性别
   - age: INT, 年龄

关系：一个学生有多条成绩记录（一对多）
查询时需要用 student_id 关联两张表。
"""

EXAMPLES = """
示例 1：
问题：张三的数学成绩是多少？
SQL：SELECT s.score FROM students st JOIN scores s ON st.id = s.student_id WHERE st.name = '张三' AND s.subject = '数学'

示例 2：
问题：高一(1)班所有学生的数学成绩
SQL：SELECT st.name, s.score FROM students st JOIN scores s ON st.id = s.student_id WHERE st.class = '高一(1)班' AND s.subject = '数学'

示例 3：
问题：数学成绩最高的学生是谁？
SQL：SELECT st.name, s.score FROM students st JOIN scores s ON st.id = s.student_id WHERE s.subject = '数学' ORDER BY s.score DESC LIMIT 1

示例 4：
问题：每个学生的平均成绩
SQL：SELECT st.name, AVG(s.score) as avg_score FROM students st JOIN scores s ON st.id = s.student_id GROUP BY st.id, st.name
"""


def generate_sql(question: str) -> list[str]:
    prompt = f"""
你是一个 MYSQL 专家。根据用户的问题、数据库表结构和示例，生成对应的 SQL 语句。

数据库表结构：
{SCHEMA}

示例：
{EXAMPLES}

用户问题：{question}

要求：
1. 只返回 JSON 格式的 SQL 字符串数组，不要其他解释
2. 根据意图生成合适的 SQL（SELECT 查询、INSERT 插入、UPDATE 更新、DELETE 删除）
3. 如果是新增成绩，学生已存在时直接用其 id 作为 student_id，不要重复插入 students 表；只有学生不存在时才先插入 students 再用 LAST_INSERT_ID()
4. 如果涉及多步操作，在数组中放多条 SQL
5. 使用正确的字段名和表名
6. 如果需要关联表，使用 JOIN

示例（返回 JSON 数组）：
["SELECT s.score FROM students st JOIN scores s ON st.id = s.student_id WHERE st.name = '张三' AND s.subject = '数学'"]

["INSERT INTO scores (student_id, subject, score, exam_type) VALUES (13, '大学英语', 88.00, '期末考试')"]

注意：如果学生已存在（如蒋娜 id=13），直接 INSERT INTO scores 并指定 student_id=13 即可，不需要再插入 students 表。
如果学生不存在，才需要先 INSERT INTO students，再用 LAST_INSERT_ID() 插入 scores。

"""

    raw = call_llm(prompt).strip()
    raw = raw.replace('```json', '').replace('```', '').strip()

    sql_list = json.loads(raw)
    if isinstance(sql_list, str):
        return [sql_list]
    return sql_list


def execute_sql(sql_list: list[str]):
    """
    遍历执行 SQL 数组
    - SELECT: 返回最后一条查询的结果和列名
    - INSERT/UPDATE/DELETE: 返回受影响的行数并自动提交
    """
    conn = pymysql.connect(**DB_CONFIG)
    try:
        last_results, last_columns = None, []

        for s in sql_list:
            with conn.cursor() as cursor:
                cursor.execute(s)
                sql_type = s.strip().upper().split()[0]

                if sql_type == 'SELECT':
                    last_results = cursor.fetchall()
                    last_columns = [desc[0] for desc in cursor.description] if cursor.description else []
                else:
                    conn.commit()
                    affected = cursor.rowcount
                    last_results = affected
                    last_columns = None

        return last_results, last_columns
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ========== 新增：结果润色 ==========
def polish_result(question: str, sql: str, results: tuple, columns: list) -> str:

    prompt = f"""
你是一个友好的助手。根据用户的问题、执行的 SQL 和查询结果，尝试对结果进一步分析或总结。

用户问题：{question}
执行的 SQL：{sql}
列名：{columns}
查询结果：{results}

要求：
1. 500字以内简洁、自然的中文回答
2. 不要解释 SQL 或技术细节
3. 直接告诉用户答案

"""
    return call_llm(prompt).strip()


def nl2sql(question: str):
    print(f"📝 用户问题：{question}")

    sql_list = generate_sql(question)
    print(f"🔧 生成的 SQL：{sql_list}")

    try:
        result, columns = execute_sql(sql_list)

        # 以最后一条 SQL 的类型决定如何展示结果
        last_sql = sql_list[-1]
        sql_type = last_sql.strip().upper().split()[0]

        if sql_type == 'SELECT':
            print(f"✅ 查询结果：{result}")
            answer = polish_result(question, last_sql, result, columns)
            print(f"💬 AI 回答：{answer}")
        else:
            affected_rows = result
            op_names = {'INSERT': '插入', 'UPDATE': '更新', 'DELETE': '删除'}
            op_name = op_names.get(sql_type, '执行')
            print(f"✅ 成功{op_name}，影响 {affected_rows} 行数据")
            answer = f"已成功{op_name} {affected_rows} 行数据。"
            print(f"💬 AI 回答：{answer}")

        return result
    except ValueError as e:
        print(f"❌ SQL 校验失败：{e}")
        return None
    except Exception as e:
        print(f"❌ 执行失败：{e}")
        return None


if __name__ == '__main__':
    # 正常查询
    # nl2sql("每个学生的平均分分别是多少分？")
    #
    # print("\n" + "=" * 50 + "\n")
    #
    # # 复杂查询（润色效果更明显）
    # nl2sql("新增学生韩雪的成绩，科目是微积分，分数92")
    # nl2sql("删除学生冯刚操作系统的成绩")
    nl2sql("修改学生郑晓操作系统的成绩为77")
    
    #
    # print("\n" + "=" * 50 + "\n")

    # 测试危险操作（会被拦截）
    # nl2sql("帮我查询一下考试作弊导致成绩归0 的同学的相关考试信息和个人信息")