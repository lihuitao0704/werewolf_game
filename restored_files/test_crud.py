import requests
import json

BASE = "http://localhost:8000"

def test(question):
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print('='*60)
    r = requests.post(f"{BASE}/nl2sql", json={"question": question})
    data = r.json()
    print(f"类型: {data.get('type')}")
    print(f"SQL: {data.get('sql')}")
    if data.get('data'):
        print(f"数据: {data.get('data')}")
    if data.get('affected_rows') is not None:
        print(f"影响行数: {data.get('affected_rows')}")
    print(f"回答: {data.get('answer')}")
    if data.get('error'):
        print(f"错误: {data.get('error')}")
    return data

print("\n" + "="*60)
print("【1. 查询 SELECT】")
print("="*60)
test("学生张明的高等数学多少分？")

print("\n" + "="*60)
print("【2. 新增 INSERT】")
print("="*60)
test("给学生李华添加一条大学英语成绩88分")

print("\n" + "="*60)
print("【3. 修改 UPDATE】")
print("="*60)
test("把张明的高等数学成绩改为95分")

print("\n" + "="*60)
print("【4. 删除 DELETE】")
print("="*60)
test("删除李华的大学英语成绩记录")

print("\n" + "="*60)
print("【5. 验证删除结果】")
print("="*60)
test("李华有哪些成绩记录？")
