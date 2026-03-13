"""
AI Agent Web界面
用Gradio把你的Agent变成一个可以在浏览器中使用的Web应用
"""
import os
import json
import requests
import chromadb
import gradio as gr
from openai import OpenAI
from pypinyin import pinyin, Style
from dotenv import load_dotenv

load_dotenv()

# ============ 配置 ============
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
MODEL = "deepseek-chat"

API_HOST = os.getenv("WEATHER_API_HOST")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
weather_headers = {"X-QW-Api-Key": WEATHER_API_KEY}


# ============ 工具函数（和之前一样） ============
def text_to_pinyin(text):
    result = pinyin(text, style=Style.NORMAL)
    return ''.join([item[0] for item in result])


def calculate(expression: str) -> str:
    try:
        result = eval(expression)
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_weather(city: str) -> str:
    pinyin_city = text_to_pinyin(city)
    get_cityid_url = f"https://{API_HOST}/geo/v2/city/lookup"
    try:
        response = requests.get(get_cityid_url, headers=weather_headers, params={"location": pinyin_city})
        response.raise_for_status()
        data = response.json()
        location_id = data['location'][0]['id']

        get_city_weather_url = f"https://{API_HOST}/v7/weather/now"
        real_weather_response = requests.get(get_city_weather_url, headers=weather_headers, params={"location": location_id})
        real_weather_response.raise_for_status()
        now = real_weather_response.json()["now"]

        weather_result = (
            f"当前天气{now['text']}，温度{now['temp']}摄氏度，体感温度{now['feelsLike']}摄氏度。"
            f"{now['windDir']}{now['windScale']}级，风速{now['windSpeed']}公里/小时。"
            f"湿度{now['humidity']}%，降水量{now['precip']}毫米，气压{now['pressure']}百帕，"
            f"能见度{now['vis']}公里，云量{now['cloud']}%，露点温度{now['dew']}摄氏度。"
        )
        return json.dumps({"weather": weather_result})
    except Exception as e:
        return json.dumps({"error": str(e)})


def search_job(job: str) -> str:
    fake_job = {
        "AI Agent开发": "岗位数量: 127个, 平均薪资: 22k, 热门公司: 华为南研所/中兴/亚信科技",
        "Java开发": "南京Java开发工程师岗位数量非常多，就业薪资一般，酌情选择",
        "数据开发": "南京数据开发工程师岗位数量一般，就业薪资一般，谨慎选择",
    }
    result = fake_job.get(job, f"{job}：暂无数据")
    return json.dumps({"job": result})


def read_knowledge(question: str) -> str:
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_collection(name="knowledge")
        results = collection.query(query_texts=[question], n_results=2)
        relevant_text = "\n\n".join(results["documents"][0])
        return json.dumps({"knowledge": relevant_text, "question": question})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============ 工具注册 ============
TOOLS_MAP = {
    "calculate": calculate,
    "get_weather": get_weather,
    "search_job": search_job,
    "read_knowledge": read_knowledge,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式，比如加减乘除、幂运算等",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "要计算的数学表达式"}
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
                    "city": {"type": "string", "description": "城市名称"}
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
                    "job": {"type": "string", "description": "岗位名称"}
                },
                "required": ["job"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_knowledge",
            "description": "从知识库中检索AI Agent相关知识",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "关于AI Agent的相关问题"}
                },
                "required": ["question"]
            }
        }
    },
]


# ============ Agent核心逻辑 ============
def run_agent(user_input: str, history: list) -> tuple:
    """
    Gradio的chatbot组件需要返回(history)格式
    history是一个列表，每个元素是{"role": "user/assistant", "content": "..."}
    """
    # 构建发送给大模型的消息
    messages = [{"role": "system", "content": "你是一个有用的助手，可以使用工具来帮助用户。回答时保持简洁。"}]

    # 把历史对话加入（只取最近10轮，避免token超限）
    for msg in history[-20:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_input})

    # 第一次调用
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS_SCHEMA,
    )

    assistant_message = response.choices[0].message
    tool_info = ""

    if assistant_message.tool_calls:
        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            tool_info += f"🔧 调用工具: {func_name}({func_args})\n"

            result = TOOLS_MAP[func_name](**func_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

        final_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        final_answer = final_response.choices[0].message.content
    else:
        final_answer = assistant_message.content

    # 如果调用了工具，在回答前面加上工具信息
    # if tool_info:
    #     display_answer = f"```\n{tool_info}```\n\n{final_answer}"
    # else:
    #     display_answer = final_answer
    display_answer = final_answer

    # 更新历史
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": display_answer})

    return history, history, ""


# ============ Gradio界面 ============
with gr.Blocks(title="AI Agent Demo") as demo:
    gr.Markdown(
        """
        # 🤖 AI Agent Demo
        **一个具备多工具调用和知识库检索能力的智能助手**

        支持的能力：🌤️ 实时天气查询 | 🔢 数学计算 | 💼 岗位信息查询 | 📚 AI知识库问答
        """
    )

    chatbot = gr.Chatbot(
        show_label=False,
    )

    state = gr.State([])

    with gr.Row():
        txt = gr.Textbox(
            show_label=False,
            placeholder="输入你的问题...（试试：南京天气怎么样 / 帮我算1024*768 / AI Agent有哪些能力）",
            scale=9,
        )
        submit_btn = gr.Button("发送", variant="primary", scale=1)

    # 绑定事件
    submit_btn.click(
        fn=run_agent,
        inputs=[txt, state],
        outputs=[chatbot, state, txt],
    )
    txt.submit(
        fn=run_agent,
        inputs=[txt, state],
        outputs=[chatbot, state, txt],
    )

demo.launch(
    theme=gr.themes.Soft(),
    css="""
    .chatbot-container { height: 500px !important; }
    footer { display: none !important; }
    """
)