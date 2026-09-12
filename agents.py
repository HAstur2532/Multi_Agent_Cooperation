import json
from cmath import nan
from typing import TypedDict, Annotated, List, Dict, Literal

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pandas._libs.tslibs.offsets import Nano

import tools


class InfoGraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    original_query: str          # 原始问题
    analysis_result: Dict        # understander输出
    chart_plan: Dict             # planner输出
    svg_code: str                # designer生成的SVG代码
    quality_report: Dict         # reviewer审查报告
    final_output: str            # reflection输出
    retry_count: int             # 重试次数
    feedback: str                # reviewer反馈

def build_agents(llm):
    understander = create_agent(
        model=llm,
        tools=[],
        system_prompt="""
        你是一位信息图内容分析师，你的任务是将用户的问题转化为结构化的信息图数据。
        请根据用户问题，输出以下JSON格式（只输出JSON，不要其他文字）：
            {
                "topic": "主题名称",
                "key_concepts": ["概念1", "概念2", "概念3"],
                "relationships": "概念之间的关系描述",
                "suggested_chart": "建议的图表类型（flowchart/timeline/comparison/hierarchy/concept_map）",
                "data_points": {"标签": "数值"}  // 如果有数值比较
            }
        主题应当简明扼要，不应该过长
        """
    )

    planner = create_agent(
        model=llm,
        tools=[],
        system_prompt="""
        你是一位信息图设计规划师。你的任务是根据分析结果以及可能存在的改进建议，设计具体的图表结构。
        请输出以下JSON格式（只输出JSON）：
            {
                "chart_type": "图表类型",
                "title": "图表标题",
                "nodes": [
                    {"id": "1", "label": "节点名称", "description": "节点描述"}
                ],
                "edges": [
                    {"from": "1", "to": "2", "label": "连接标签"}
                ],
                "layout": "vertical"  // 或 horizontal
            }
        
        对于时间线，节点格式：{"id": "1", "label": "年份", "description": "事件"}
        对于对比图，节点格式：{"id": "1", "label": "名称", "value": 数值}
        对于层次图，节点格式：{"id": "1", "label": "名称", "level": 层级数字}
        
        支持的图表类型：
            - flowchart: 流程图
            - timeline: 时间线
            - comparison: 对比柱状图
            - hierarchy: 层次结构图
            - concept_map: 概念关系图
            
        对于流程图，节点描述需要精炼，控制在30字符以内。
        
        你的输出必须是且仅是一个合法的JSON对象，且尽可能满足反馈的改进建议。
        """
    )

    designer = create_agent(
        model=llm,
        tools=[tools.generate_infographic],
        system_prompt=""""
        你是一位SVG信息图设计师。你的任务是根据规划生成高质量的SVG代码。
        工作流程：
            1. 将规划JSON传递给 generate_infographic 工具，生成SVG代码
            2. 检查生成的SVG是否正确
            3. 返回SVG代码和保存路径
        """

    )

    reviewer = create_agent(
        model=llm,
        tools=[tools.validate_svg],
        system_prompt="""
        你是一位SVG质量审查员。你的任务是对生成的SVG进行全面检查。
        
        审查维度：
            1. **格式正确性**：SVG 标签是否闭合、是否可渲染
            2. **内容相关性**：图表内容是否与用户问题相关
            3. **信息完整性**：关键概念是否都覆盖了
            4. **结构合理性**：图表类型选择是否合适
            5. **可读性**：文字是否清晰、布局是否合理
            
        你必须首先调用 validate_svg 工具检查SVG的格式正确性，然后基于工具返回的结果和你自己对SVG代码内容的理解，给出综合评分。

        工作流程：
            1. 调用 validate_svg 工具检查SVG代码格式
            2. 基于SVG代码，从中提取文本评估其余维度
            3. 根据工具返回的结果及你的评估，输出以下JSON格式：
                {
                    "score": 5,           // 1-5分
                    "issues": ["问题1"],  // 如果有问题
                    "suggestion": "具体的改进建议",
                    "content_summary": "SVG图的主要内容"
                }
        
        如果代码不完整或缺失，请报告问题。
        除JSON数据外，不能输入任何内容。
        """
    )

    return {
        "understander": understander,
        "planner": planner,
        "designer": designer,
        "reviewer": reviewer
    }


def create_node(agent, extract_field: str = None, inject_fields: List[str] = None, replacable: bool = False):
    """将智能体包装成LangGraph节点"""

    def node(state: InfoGraphState):
        messages = state.get("messages", [])
        if not messages:
            return {}

        injected_parts = []
        if inject_fields:
            for field in inject_fields:
                data = state.get(field)
                if data:
                    if isinstance(data, dict):
                        injected_parts.append(json.dumps(data, ensure_ascii=False, indent=2))
                    else:
                        injected_parts.append(str(data))

        if injected_parts:
            inject_content = "\n\n".join(injected_parts)
            if replacable:
                messages = [HumanMessage(content=f"请根据以下信息设计图表规划（只输出JSON）：\n{inject_content}")]
            else:
                messages = list(messages) + [HumanMessage(content=f"参考信息：\n{inject_content}")]

        result = agent.invoke({"messages": messages})
        last_msg = result["messages"][-1]

        # 尝试解析JSON
        try:
            content = last_msg.content
            # 提取可能的JSON块
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            data = json.loads(content)
            return {extract_field: data} if extract_field else {"messages": [AIMessage(content=last_msg.content)]}
        except:
            return {"messages": [AIMessage(content=last_msg.content)]}

    return node


def designer_node(state: InfoGraphState):
    """设计师节点：直接从状态读取 chart_plan，调用工具生成 SVG"""
    chart_plan = state.get("chart_plan", {})

    if not chart_plan:
        return {
            "messages": [AIMessage(content="缺少图表规划数据，无法生成SVG。")],
            "svg_code": ""
        }

    try:
        # 直接调用工具函数
        plan_json = json.dumps(chart_plan, ensure_ascii=False)
        svg_code = tools.generate_infographic.invoke({"plan_json": plan_json})

        if not svg_code or not svg_code.strip().startswith("<svg"):
            return {
                "messages": [AIMessage(content=f"SVG生成失败：{svg_code}")],
                "svg_code": ""
            }

        return {
            "messages": [AIMessage(content=f"以下是生成的SVG代码：\n\n{svg_code}")],
            "svg_code": svg_code
        }
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"designer出错：{str(e)}")],
            "svg_code": ""
        }

def build_multi_agent_graph(llm):
    agents = build_agents(llm)

    def reflection_node(state: InfoGraphState):
        report = state.get("quality_report", {})
        score = report.get("score", 5)
        retry_count = state.get("retry_count", 0)

        if score >= 4:
            return {
                "final_output": "通过",
                "messages": [AIMessage(content=f"质量达标（{score}/5）。")]
            }

        if retry_count >= 2:
            return {
                "final_output": "通过",
                "messages": [AIMessage(content=f"重试次数已达上限（{retry_count}次），使用当前版本。")]
            }

        suggestion = report.get("suggestion", "请优化图表结构")

        return {
            "final_output": "重试",
            "feedback": f"审查反馈：{suggestion}",  # ← 写入状态
            "retry_count": retry_count + 1,
            "messages": [AIMessage(content=f"第 {retry_count + 1} 次重试，已将反馈传递给planner。")]
        }
    workflow = StateGraph(InfoGraphState)

    # 添加节点
    workflow.add_node("understander", create_node(agents["understander"], "analysis_result"))
    workflow.add_node("planner", create_node(agents["planner"], extract_field="chart_plan", inject_fields=["analysis_result", "feedback"], replacable=True))
    workflow.add_node("designer", designer_node)
    workflow.add_node("reviewer", create_node(agents["reviewer"], "quality_report"))
    workflow.add_node("reflection", reflection_node)


    # 设置入口
    workflow.set_entry_point("understander")

    # 顺序执行
    workflow.add_edge("understander", "planner")
    workflow.add_edge("planner", "designer")
    workflow.add_edge("designer", "reviewer")
    workflow.add_edge("reviewer", "reflection")

    def route_after_reflection(state: InfoGraphState) -> Literal["planner", END]:
        report = state.get("quality_report", {})
        score = report.get("score", 5)
        return "planner" if state.get("final_output") == "重试" else END

    workflow.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {"planner": "planner", END: END}
    )

    return workflow.compile()


def run_with_trace(graph, user_query: str):
    """运行智能体并打印详细轨迹"""
    print(f"用户: {user_query}\n")

    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "original_query": user_query,
        "analysis_result": {},
        "chart_plan": {},
        "svg_code": "",
        "quality_report": {},
        "final_output": "",
        "feedback": ""
    }

    step_names = {
        "understander": "[understander]",
        "planner": "[planner]",
        "designer": "[designer]",
        "reviewer": "[reviewer]",
        "reflection": "[reflection]"
    }

    final_state = initial_state.copy()

    for step in graph.stream(initial_state, stream_mode="updates"):
        for node_name, update in step.items():
            if node_name in step_names:
                print(f"\n{step_names[node_name]}:")

                if "analysis_result" in update:
                    data = update["analysis_result"]
                    print(f"  主题: {data.get('topic', '未识别')}")
                    print(f"  建议图表: {data.get('suggested_chart', '未指定')}")

                if "chart_plan" in update:
                    data = update["chart_plan"]
                    print(f"  图表类型: {data.get('chart_type', '未指定')}")
                    print(f"  节点数: {len(data.get('nodes', []))}")

                if "messages" in update:
                    last = update["messages"][-1]
                    if hasattr(last, "content") and last.content:
                        content = last.content
                        if "svg" in content.lower() and "<svg" in content:
                            print(f"  SVG代码: 已生成 ({len(content)} 字符)")
                        else:
                            print(f"  输出: {content}")

                if "quality_report" in update:
                    data = update["quality_report"]
                    print(f"得分：{data.get('score', nan)}/5")
                    print(f"问题: {data.get('issues', '无')}")
                    print(f"建议：{data.get('suggestion', '无')}")

                if "final_output" in update:
                    print(f"  决策: {update.get('final_output')}")
                    if update.get('final_output') == "重试":
                        print(f"  重试次数: {update.get('retry_count', 0)}")

                # 保存最终状态
                final_state.update(update)

    svg_code = final_state.get("svg_code", "")

    if svg_code and svg_code.strip().startswith("<svg"):
        print("\n信息图生成完成！")
        # 保存文件
        save_result = tools.save_svg_to_file.invoke({"svg_code": svg_code})
        print(save_result)
    else:
        print("未检测到有效的SVG代码")
        # 打印调试信息
        print(f"   state keys: {list(final_state.keys())}")
        print(f"   svg_code length: {len(svg_code)}")

    return final_state

