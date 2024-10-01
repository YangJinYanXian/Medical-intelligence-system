from flask import Flask
# 创建要给app对象
app = Flask(__name__)

# 使用装饰器定义接口函数
# 当用户访问hello时，就会调用下面的hello_world函数
@app.route("/hello")
def hello_world():
    # 直接返回字符串
    return "Hello World!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) # 5000端口，需要把远程服务的5000端口打开




