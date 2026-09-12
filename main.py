import json
import pandas as pd
from langchain_openai import ChatOpenAI
import tools
import agents

try:
    id = "6081486"
    df = pd.read_csv("data/默认业务空间-apiKey-6081486.csv")
    api_key = df[id].values[0]
    api_host = df[id].values[2]
except Exception as e:
    print(e)


# LLM配置
llm = ChatOpenAI(
    model="qwen3.6-flash-2026-04-16",
    openai_api_key=api_key,
    openai_api_base=api_host,
    temperature=0
)

tools.inject_llm(llm)

graph = agents.build_multi_agent_graph(llm)

questions = [# "大语言模型的基本原理",
             # "通俗易懂地解释词向量（Word Embedding）的基本概念",
             "中山大学的发展历程",
             # " 绘制 SVG 流程图，展示从一颗咖啡豆到一杯咖啡的完整生产链（种植、采摘、烘焙、研磨、冲煮）",
             # "YouTube has 10 times more videos than TikTok, TikTok has 2 times more than Kuaishou",
             # "台风从形成到登录的过程是什么",
             # "多智能体协作的过程"
             ]

for q in questions:
    agents.run_with_trace(graph, q)
