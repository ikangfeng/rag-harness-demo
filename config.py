import os


class rag_test_Config():
    # DashScope API Key 从环境变量读取，真实 Key 不要提交到 Git
    # 本地/服务器启动前：export DASHSCOPE_API_KEY="sk-xxxx"
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    elastic_password = ''
    elastic_username = ""
    elastic_host = ""
    server_host = ""
