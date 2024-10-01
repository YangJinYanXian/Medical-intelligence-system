from neo4j import GraphDatabase

uri = "bolt://192.168.88.161:7687"

driver = GraphDatabase.driver(uri, auth=("neo4j", "123456") # 密码自己修改的
                              , max_connection_lifetime=1000)

# 创建一个会话
with driver.session() as session:
    cypher = "create (c:Company) SET c.name='黑马程序员' return c.name"
    record = session.run(cypher)
    print(type(record), record)#, list(record))
    result = list(map(lambda x: x[0], record))
    print("result: ", result)
