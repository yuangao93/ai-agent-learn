"""
你的第一个AI Agent
它和普通聊天机器人的区别：它能使用工具
"""
import os
import json
import requests
import chromadb
from openai import OpenAI
from pypinyin import pinyin, Style
from dotenv import load_dotenv

load_dotenv()


# ============ 配置 ============
client = OpenAI(
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = "https://api.deepseek.com"
)
MODEL = "deepseek-chat"

API_HOST = os.getenv("WEATHER_API_HOST")
API_KEY = os.getenv("WEATHER_API_KEY")

weather_headers = {
    "X-QW-Api-Key": API_KEY
}


# ============ 定义工具 ============
# 这就是Agent的核心：让AI能调用外部函数

#将文字转成拼音并且拼接在一起
def text_to_pinyin(text):
    # 返回结果是列表套列表格式，如 [['nan'], ['jing']]，需拼接处理
    result = pinyin(text, style=Style.NORMAL)
    return ''.join([item[0] for item in result])

def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_weather(city: str) -> str:
    """获取天气信息（模拟数据，后面会换成真实API）
    fake_weather = {
        "南京": "晴天，18°C，适合出门",
        "北京": "多云，15°C，有轻度雾霾",
        "上海": "小雨，16°C，记得带伞",
    }
    result = fake_weather.get(city, f"{city}：暂无数据")
    下面使用和风天气api获取真实天气数据
    """
    pinyin_city = text_to_pinyin(city)
    get_cityid_url = "https://"+API_HOST+"/geo/v2/city/lookup"
    try:
        response = requests.get(get_cityid_url, headers=weather_headers,params={"location":pinyin_city})
        response.raise_for_status()
        # 解析JSON数据
        data = response.json()
        location_id = data['location'][0]['id']
        print("请求成功，返回数据：",location_id)

        # 根据得到的location_id,再次发请求
        get_city_weather_url = "https://"+API_HOST+"/v7/weather/now"
        real_weather_response = requests.get(get_city_weather_url, headers=weather_headers,params={"location":location_id})
        real_weather_response.raise_for_status()
        # 解析JSON数据
        weather_data = real_weather_response.json()
        now = weather_data["now"]
        # 提取所需字段
        temp = now['temp']
        feels_like = now['feelsLike']
        weather_text = now['text']
        wind_dir = now['windDir']
        wind_scale = now['windScale']
        wind_speed = now['windSpeed']
        humidity = now['humidity']
        precip = now['precip']
        pressure = now['pressure']
        vis = now['vis']
        cloud = now['cloud']
        dew = now['dew']

        # 拼接成完整语句
        weather_result = (f"当前天气{weather_text}，温度{temp}摄氏度，体感温度{feels_like}摄氏度。"
                    f"{wind_dir}{wind_scale}级，风速{wind_speed}公里/小时。"
                    f"湿度{humidity}%，降水量{precip}毫米，气压{pressure}百帕，"
                    f"能见度{vis}公里，云量{cloud}%，露点温度{dew}摄氏度。")

    except requests.exceptions.RequestException as e:
        print(f"请求发生错误：{e}")


    return json.dumps({"weather": weather_result})

def search_job(job: str) -> str:
    """获取岗位数据（模拟数据，后面会换成真实API）"""
    fake_job = {
        "AI Agent开发": "岗位数量: 127个, 平均薪资: 22k, 热门公司: 华为南研所/中兴/亚信科技",
        "Java开发": "南京Java开发工程师岗位数量非常多，就业薪资一般，酌情选择",
        "数据开发": "南京数据开发工程师岗位数量一般，就业薪资一般，谨慎选择",
    }
    result = fake_job.get(job, f"{job}：暂无数据")
    return json.dumps({"job": result})

# 从本地的一个txt文件中读取内容，把内容和问题一起发给大模型
# def read_knowledge(question: str) -> str:
#     with open("knowledge.txt", "r", encoding="utf-8") as f:
#         knowledge = f.read()
#     return json.dumps({"knowledge": knowledge, "question": question})

# 从向量知识库中检索最相关的内容来回答问题
def read_knowledge(question: str) -> str:
    try:
        chromadb_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chromadb_client.get_collection(name="knowledge")

        results = collection.query(
            query_texts=[question],
            n_results=2
        )

        # 把检索到的段落拼接起来
        relevant_text = "\n\n".join(results["documents"][0])

        return json.dumps({"knowledge": relevant_text, "question": question})
    except Exception as e:
        return json.dumps({"error": str(e)})


# 工具注册表：Agent通过这个知道自己有哪些工具可以用
TOOLS_MAP = {
    "calculate": calculate,
    "get_weather": get_weather,
    "search_job": search_job,
    "read_knowledge": read_knowledge,
}

# 告诉大模型有哪些工具可以调用（这是OpenAI的标准格式，DeepSeek兼容）
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式，比如加减乘除、幂运算等",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，比如 '2+3*4' 或 '100/7'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，比如'南京'、'北京'"
                    }
                },
                "required": ["city"]
            }
        }
    },
{
        "type": "function",
        "function": {
            "name": "search_job",
            "description": "获取指定城市的工作岗位信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "job": {
                        "type": "string",
                        "description": "岗位名称，比如：'AI Agent开发'、'Java开发'、'数据开发'"
                    }
                },
                "required": ["job"]
            }
        }
    },
{
        "type": "function",
        "function": {
            "name": "read_knowledge",
            "description": "从项目根目录下读取文件中的知识作为知识库",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "关于AI agent的相关问题"
                    }
                },
                "required": ["question"]
            }
        }
    },
]

# ============ 全局对话历史 ============
conversation_history = [
    {"role": "system", "content": "你是一个有用的助手，可以使用工具来帮助用户。"}
]

# ============ Agent核心循环 ============

def run_agent(user_input: str):
    """
    Agent的核心逻辑，就这么简单：
    1. 把用户的问题和工具列表一起发给大模型
    2. 如果大模型决定调用工具，就执行工具并把结果返回给大模型
    3. 大模型根据工具结果生成最终回答
    4. 如果大模型决定不调用工具，就直接回答
    """
    print(f"\n{'='*50}")
    print(f"你: {user_input}")
    print(f"{'='*50}")

    # messages = [
    #     {"role": "system", "content": "你是一个有用的助手，可以使用工具来帮助用户。"},
    #     {"role": "user", "content": user_input}
    # ]
    # 将用户输入添加到对话历史
    conversation_history.append({"role": "user", "content": user_input})

    # 第一次调用：让大模型决定要不要用工具
    response = client.chat.completions.create(
        model=MODEL,
        messages=conversation_history,
        tools=TOOLS_SCHEMA,
    )

    assistant_message = response.choices[0].message

    # 检查大模型是否决定调用工具
    if assistant_message.tool_calls:
        print(f"\n[Agent思考] 我需要使用工具来回答这个问题")

        # 把助手的回复加入对话历史
        conversation_history.append(assistant_message)

        # 执行每一个工具调用
        for tool_call in assistant_message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"[Agent行动] 调用工具: {func_name}，参数: {func_args}")

            # 执行工具函数
            result = TOOLS_MAP[func_name](**func_args)
            print(f"[工具结果] {result}")

            # 把工具结果加入对话历史
            conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

        # 第二次调用：让大模型根据工具结果生成最终回答
        final_response = client.chat.completions.create(
            model=MODEL,
            messages=conversation_history,
        )
        final_answer = final_response.choices[0].message.content
        # 把最终回答也加入历史
        conversation_history.append({"role": "assistant", "content": final_answer})
    else:
        # 大模型决定不用工具，直接回答
        print(f"\n[Agent思考] 这个问题我可以直接回答")
        final_answer = assistant_message.content
        # 把最终回答也加入历史
        conversation_history.append({"role": "assistant", "content": final_answer})

    print(f"\nAgent: {final_answer}")
    return final_answer

# ============ 运行 ============
if __name__ == "__main__":
    print("=" * 50)
    print("  你的第一个AI Agent已启动！")
    print("  试试问我：")
    print("    - 南京今天天气怎么样？")
    print("    - 帮我算一下 1024 * 768")
    print("    - 南京AI Agent开发岗位怎么样？")
    print("    - 你是谁？（不需要工具的问题）")
    print("  输入 quit 退出")
    print("=" * 50)

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("再见！")
            break
        if not user_input:
            continue
        run_agent(user_input)