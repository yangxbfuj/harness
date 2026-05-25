from pathlib import Path

from langchain.tools import tool

@tool
def execute_python(code: str) -> str:
    """执行Python代码并返回结果。"""
    try:
        # 创建本地环境执行代码
        local_vars = {}
        exec(code, {}, local_vars)  # python可以动态 执行 代码
        result= local_vars.get('result', '执行成功')
        print("##执行结果:\n",result)
        return str(result)
    except Exception as e:
        return f"Error executing code: {str(e)}"


def current_path() -> str:
    """获取当前文件目录"""
    return Path(__file__).parent.resolve()

tools = [execute_python]