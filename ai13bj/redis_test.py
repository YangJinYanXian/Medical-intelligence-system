# redis配置
REDIS_CONFIG = {
    "host": "0.0.0.0",
    "port": 6379
}

import redis

# 创建redis连接池
pool = redis.ConnectionPool(**REDIS_CONFIG)

# 从连接池中获取连接对象
r = redis.StrictRedis(connection_pool=pool)

# 进行数据存取操作
# 存数据
uid = "88888"
# key 数据的描述信息
key = "最后一句话:".encode('utf-8')
# value 具体存储的数据
value = "有点头疼".encode('utf-8')

# 存数据
r.hset(uid, key, value)

# 取数据
result = r.hget(uid, key)
print(result.decode('utf-8'))


