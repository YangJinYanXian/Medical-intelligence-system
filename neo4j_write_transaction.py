from neo4j import GraphDatabase

uri = "bolt://192.168.88.161:7687"

driver = GraphDatabase.driver(uri, auth=("neo4j", "123456") # 密码自己修改的
                              , max_connection_lifetime=1000)

def _some_operations(tx, cat_name, mouse_name):
    tx.run("merge (a:Cat{name: $cat_name})"
           "merge (b:Mouse{name: $mouse_name})"
           "merge (a)-[r:And]-(b)",
           cat_name=cat_name, mouse_name=mouse_name
           )
def _some_operations1(tx, cat_name, mouse_name):
    tx.run("merge (a:Cat{name: $cat_name})"
           "merge (b:Mouse{name: $mouse_name})"
           "create (a)-[r:And]-(b)",
           cat_name=cat_name, mouse_name=mouse_name
           )

with driver.session() as session:
    session.write_transaction(_some_operations1, "Tom1", "Jerry1")