import json
import math
import os
from datetime import datetime
from typing import List, Dict, Any

from langchain_core.tools import tool

_llm = None

def inject_llm(llm_instance):
    global _llm
    _llm = llm_instance

COLORS = {
    "primary": "#4A90D9",
    "secondary": "#67B7DC",
    "accent": "#F5A623",
    "success": "#7ED321",
    "danger": "#D0021B",
    "warning": "#F8E71C",
    "dark": "#2C3E50",
    "light": "#ECF0F1",
    "white": "#FFFFFF",
    "text": "#333333",
    "border": "#BDC3C7"
}

FONT_CONFIG = {
    "family": "Arial, 'Microsoft YaHei', sans-serif",
    "title_size": 24,
    "heading_size": 18,
    "body_size": 14,
    "small_size": 12
}


@tool
def generate_infographic(plan_json: str) -> str:
    """
    根据图表规划JSON生成SVG信息图。
    输入参数：plan_json - 包含chart_type、title、nodes等字段的JSON字符串。
    返回：完整的SVG代码字符串。

    支持的图表类型：
    - flowchart: 流程图
    - timeline: 时间线
    - comparison: 对比柱状图
    - hierarchy: 层次结构图
    - concept_map: 概念关系图
    """
    try:
        plan = json.loads(plan_json)
        chart_type = plan.get("chart_type", "flowchart")

        # 路由到对应的生成函数
        if chart_type == "flowchart":
            return _generate_flowchart(plan)
        elif chart_type == "timeline":
            return _generate_timeline(plan)
        elif chart_type == "comparison":
            return _generate_comparison(plan)
        elif chart_type == "hierarchy":
            return _generate_hierarchy(plan)
        elif chart_type == "concept_map":
            return _generate_concept_map(plan)
        else:
            return _generate_error_svg(f"不支持的图表类型: {chart_type}")

    except json.JSONDecodeError as e:
        return _generate_error_svg(f"JSON解析失败: {str(e)}")
    except Exception as e:
        return _generate_error_svg(f"生成SVG时出错: {str(e)}")


@tool
def save_svg_to_file(svg_code: str, filename: str = None, output_dir: str = "output") -> str:
    """
    将SVG代码保存为本地文件。
    输入参数：
    - svg_code: SVG代码字符串
    - filename: 文件名
    - output_dir: 输出目录（默认"output"）
    返回：保存成功后的文件路径
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"infographic_{timestamp}.svg"

        if not filename.endswith(".svg"):
            filename += ".svg"

        filepath = os.path.join(output_dir, filename)

        # 写入文件
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(svg_code)

        return f"SVG已保存至: {filepath}"

    except Exception as e:
        return f"保存失败: {str(e)}"


@tool
def validate_svg(svg_code: str) -> str:
    """
    验证SVG代码的完整性，返回质量评分和问题列表。
    输入参数：svg_code - SVG代码字符串
    返回：质量评估报告
    """
    issues = []
    score = 5

    if not svg_code or len(svg_code.strip()) < 10:
        return "SVG代码为空或过短"

    if not svg_code.strip().startswith("<svg"):
        issues.append("缺少<svg>起始标签")
        score -= 2
    if not svg_code.strip().endswith("</svg>"):
        issues.append("缺少</svg>结束标签")
        score -= 2
    if "xmlns" not in svg_code:
        issues.append("缺少xmlns命名空间声明")
        score -= 1
    if "<text" not in svg_code:
        issues.append("没有文本内容")
        score -= 1
    if "fill" not in svg_code:
        issues.append("缺少颜色填充属性")
        score -= 1

    score = max(score, 0)

    if issues:
        return f"质量评分: {score}/5\n问题:\n" + "\n".join(f"  - {issue}" for issue in issues)
    else:
        return f"质量评分: {score}/5\nSVG代码结构完整，可以正常渲染。"


def _generate_flowchart(plan: Dict[str, Any]) -> str:
    """生成纵向流程图"""
    title = plan.get("title", "流程图")
    nodes = plan.get("nodes", [])
    layout = plan.get("layout", "vertical")

    if not nodes:
        return _generate_error_svg("流程图缺少节点数据")

    node_width = 220
    node_height = 55
    spacing = 30
    padding = 50

    if layout == "vertical":
        total_height = padding * 2 + len(nodes) * (node_height + spacing) - spacing + 60
        total_width = padding * 2 + node_width + 40
        svg = _build_svg_header(total_width, total_height, title)
        y = padding + 40
        for i, node in enumerate(nodes):
            x = (total_width - node_width) / 2
            color = COLORS["primary"] if i % 2 == 0 else COLORS["secondary"]
            svg += _draw_rounded_rect(x, y, node_width, node_height, color,
                                      node.get("label", f"步骤{i + 1}"),
                                      node.get("description", ""))
            if i < len(nodes) - 1:
                svg += _draw_arrow(total_width / 2, y + node_height,
                                   total_width / 2, y + node_height + spacing, "↓")
            y += node_height + spacing
    else:
        total_width = padding * 2 + len(nodes) * (node_width + spacing) - spacing
        total_height = padding * 2 + node_height + 80
        svg = _build_svg_header(total_width, total_height, title)
        x = padding
        y = padding + 40
        for i, node in enumerate(nodes):
            color = COLORS["primary"] if i % 2 == 0 else COLORS["secondary"]
            svg += _draw_rounded_rect(x, y, node_width, node_height, color,
                                      node.get("label", f"步骤{i + 1}"),
                                      node.get("description", ""))
            if i < len(nodes) - 1:
                svg += _draw_arrow(x + node_width, y + node_height / 2,
                                   x + node_width + spacing, y + node_height / 2, "→")
            x += node_width + spacing

    svg += "</svg>"
    return svg


def _generate_timeline(plan: Dict[str, Any]) -> str:
    """生成时间线图"""
    title = plan.get("title", "时间线")
    nodes = plan.get("nodes", [])

    if not nodes:
        return _generate_error_svg("时间线缺少节点数据")

    padding = 50
    node_spacing = 70
    total_height = padding * 2 + len(nodes) * node_spacing + 40
    total_width = 700
    line_x = 120

    svg = _build_svg_header(total_width, total_height, title)

    # 时间线主轴
    svg += f'  <line x1="{line_x}" y1="{padding + 20}" x2="{line_x}" y2="{total_height - padding - 20}" stroke="{COLORS["dark"]}" stroke-width="3"/>\n'
    # 时间线起点和终点的小圆点
    svg += f'  <circle cx="{line_x}" cy="{padding + 20}" r="6" fill="{COLORS["primary"]}"/>\n'
    svg += f'  <circle cx="{line_x}" cy="{total_height - padding - 20}" r="6" fill="{COLORS["primary"]}"/>\n'

    y = padding + 30
    for i, node in enumerate(nodes):
        # 时间点圆点
        svg += f'  <circle cx="{line_x}" cy="{y}" r="10" fill="{COLORS["primary"]}" stroke="{COLORS["white"]}" stroke-width="3"/>\n'
        if i < len(nodes) - 1:
            svg += f'  <line x1="{line_x}" y1="{y + 10}" x2="{line_x}" y2="{y + node_spacing - 10}" stroke="{COLORS["primary"]}" stroke-width="3" stroke-dasharray="5,5"/>\n'
        # 年份标签（左侧）
        label = node.get("label", "")
        svg += f'  <text x="{line_x - 20}" y="{y + 5}" font-family="{FONT_CONFIG["family"]}" font-size="16" fill="{COLORS["dark"]}" text-anchor="end" font-weight="bold">{label}</text>\n'
        # 描述（右侧）
        desc = node.get("description", "")
        if len(desc) > 40:
            desc = desc[:38] + "..."
        svg += f'  <text x="{line_x + 25}" y="{y + 5}" font-family="{FONT_CONFIG["family"]}" font-size="14" fill="{COLORS["text"]}" text-anchor="start">{desc}</text>\n'
        y += node_spacing

    svg += "</svg>"
    return svg


def _generate_comparison(plan: Dict[str, Any]) -> str:
    """生成对比柱状图"""
    title = plan.get("title", "对比图")
    nodes = plan.get("nodes", [])

    if not nodes:
        return _generate_error_svg("对比图缺少数据")

    # 提取数值
    max_value = max([n.get("value", 1) for n in nodes]) * 1.3
    padding = 60
    chart_height = 280
    chart_width = 500
    bar_width = 55
    bar_spacing = 35
    bottom_margin = 50

    total_width = max(chart_width, padding * 2 + len(nodes) * (bar_width + bar_spacing))
    total_height = padding + chart_height + bottom_margin + 60

    svg = _build_svg_header(total_width, total_height, title)

    # 坐标轴
    axis_y = padding + chart_height
    svg += f'  <line x1="{padding - 10}" y1="{axis_y}" x2="{total_width - padding + 10}" y2="{axis_y}" stroke="{COLORS["dark"]}" stroke-width="2"/>\n'
    svg += f'  <line x1="{padding - 10}" y1="{padding}" x2="{padding - 10}" y2="{axis_y}" stroke="{COLORS["dark"]}" stroke-width="2"/>\n'

    # Y轴刻度
    for i in range(5):
        val = int(max_value / 4 * i)
        y = axis_y - (val / max_value) * chart_height
        if val > 0:
            svg += f'  <line x1="{padding - 25}" y1="{y}" x2="{padding - 10}" y2="{y}" stroke="{COLORS["border"]}" stroke-width="1"/>\n'
            svg += f'  <text x="{padding - 30}" y="{y + 4}" font-family="{FONT_CONFIG["family"]}" font-size="11" fill="{COLORS["text"]}" text-anchor="end">{val}</text>\n'

    x = padding + bar_spacing / 2
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], COLORS["success"], COLORS["danger"]]

    for i, node in enumerate(nodes):
        value = node.get("value", 0)
        label = node.get("label", "")
        bar_height = (value / max_value) * chart_height
        y = axis_y - bar_height
        color = colors[i % len(colors)]

        # 柱子
        svg += f'  <rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" rx="4" fill="{color}" opacity="0.85"/>\n'
        # 数值标签
        svg += f'  <text x="{x + bar_width / 2}" y="{y - 8}" font-family="{FONT_CONFIG["family"]}" font-size="14" fill="{COLORS["dark"]}" text-anchor="middle" font-weight="bold">{value}</text>\n'
        # 底部标签
        svg += f'  <text x="{x + bar_width / 2}" y="{axis_y + 25}" font-family="{FONT_CONFIG["family"]}" font-size="13" fill="{COLORS["text"]}" text-anchor="middle">{label}</text>\n'
        x += bar_width + bar_spacing

    svg += "</svg>"
    return svg


def _generate_hierarchy(plan: Dict[str, Any]) -> str:
    """生成层次结构图"""
    title = plan.get("title", "层次结构图")
    nodes = plan.get("nodes", [])
    edges = plan.get("edges", [])

    if not nodes:
        return _generate_error_svg("层次图缺少节点数据")

    # 按层级分组
    levels = {}
    for node in nodes:
        level = node.get("level", 0)
        if level not in levels:
            levels[level] = []
        levels[level].append(node)

    max_level = max(levels.keys()) if levels else 0
    level_height = 80
    node_width = 150
    node_height = 42
    padding = 50

    max_nodes_in_level = max([len(nodes) for nodes in levels.values()], default=1)
    total_width = max(600, max_nodes_in_level * (node_width + 30) + padding * 2)
    total_height = padding * 2 + (max_level + 1) * (level_height + 20) + 50

    svg = _build_svg_header(total_width, total_height, title)

    # 存储节点坐标供连线使用
    node_positions = {}

    for level, level_nodes in levels.items():
        y = padding + 40 + level * (level_height + 20)
        x_positions = _distribute_nodes(len(level_nodes), total_width, padding, node_width)

        for i, node in enumerate(level_nodes):
            x = x_positions[i]
            node_id = node.get("id", f"node_{level}_{i}")
            node_positions[node_id] = {"x": x + node_width / 2, "y": y + node_height}

            color = COLORS["primary"] if level == 0 else COLORS["secondary"] if level == 1 else COLORS["accent"]
            svg += _draw_rounded_rect(x, y, node_width, node_height, color,
                                      node.get("label", ""), "")

    # 绘制连接线
    for edge in edges:
        from_id = edge.get("from")
        to_id = edge.get("to")
        if from_id in node_positions and to_id in node_positions:
            from_pos = node_positions[from_id]
            to_pos = node_positions[to_id]
            svg += _draw_bezier(from_pos["x"], from_pos["y"],
                                to_pos["x"], to_pos["y"] - node_height, COLORS["border"])

    svg += "</svg>"
    return svg


def _generate_concept_map(plan: Dict[str, Any]) -> str:
    """生成概念关系图"""
    title = plan.get("title", "概念关系图")
    nodes = plan.get("nodes", [])
    edges = plan.get("edges", [])

    if not nodes:
        return _generate_error_svg("概念图缺少节点数据")

    total_width = 700
    total_height = 480
    padding = 40

    svg = _build_svg_header(total_width, total_height, title)

    # 自动布局：圆形分布
    radius = min(total_width, total_height) * 0.33
    center_x = total_width / 2
    center_y = total_height / 2 + 10

    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / len(nodes) - math.pi / 2
        node["x"] = center_x + radius * math.cos(angle)
        node["y"] = center_y + radius * math.sin(angle)

    # 绘制连接线
    for edge in edges:
        from_node = next((n for n in nodes if n.get("id") == edge.get("from")), None)
        to_node = next((n for n in nodes if n.get("id") == edge.get("to")), None)
        if from_node and to_node:
            x1, y1 = from_node.get("x", 0), from_node.get("y", 0)
            x2, y2 = to_node.get("x", 0), to_node.get("y", 0)
            svg += f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{COLORS["border"]}" stroke-width="2" stroke-dasharray="5,5"/>\n'
            if edge.get("label"):
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2 - 12
                svg += f'  <text x="{mid_x}" y="{mid_y}" font-family="{FONT_CONFIG["family"]}" font-size="12" fill="{COLORS["text"]}" text-anchor="middle" font-style="italic">{edge["label"]}</text>\n'

    # 绘制节点
    for node in nodes:
        x, y = node.get("x", 0), node.get("y", 0)
        label = node.get("label", "")
        node_w = max(100, len(label) * 12 + 30)
        svg += f'  <rect x="{x - node_w / 2}" y="{y - 20}" width="{node_w}" height="40" rx="20" fill="{COLORS["primary"]}" opacity="0.9" stroke="{COLORS["dark"]}" stroke-width="2"/>\n'
        svg += f'  <text x="{x}" y="{y + 6}" font-family="{FONT_CONFIG["family"]}" font-size="14" fill="{COLORS["white"]}" text-anchor="middle" font-weight="bold">{label}</text>\n'

    svg += "</svg>"
    return svg

def _build_svg_header(width: int, height: int, title: str) -> str:
    """构建SVG头部"""
    return f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:{COLORS["light"]}; font-family:{FONT_CONFIG["family"]};">
  <rect width="100%" height="100%" fill="{COLORS["light"]}" rx="10"/>
  <text x="{width/2}" y="32" font-size="{FONT_CONFIG["title_size"]}" fill="{COLORS["dark"]}" text-anchor="middle" font-weight="bold">{title}</text>
'''

def _draw_rounded_rect(x: float, y: float, w: float, h: float, color: str, label: str, desc: str) -> str:
    """绘制圆角矩形节点"""
    svg = f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{color}" stroke="{COLORS["dark"]}" stroke-width="2"/>\n'
    svg += f'  <text x="{x + w/2}" y="{y + h/2 + 4}" font-size="{FONT_CONFIG["body_size"]}" fill="{COLORS["white"]}" text-anchor="middle" font-weight="bold">{label}</text>\n'
    if desc and len(desc) > 0:
        svg += f'  <text x="{x + w/2}" y="{y + h - 10}" font-size="{FONT_CONFIG["small_size"]}" fill="{COLORS["white"]}" text-anchor="middle" opacity="0.8">{desc[:20]}</text>\n'
    return svg

def _draw_arrow(x1: float, y1: float, x2: float, y2: float, symbol: str) -> str:
    """绘制箭头"""
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    return f'''  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{COLORS["dark"]}" stroke-width="2.5"/>
  <text x="{mid_x}" y="{mid_y - 12}" font-size="20" fill="{COLORS["dark"]}" text-anchor="middle">{symbol}</text>
'''

def _draw_bezier(x1: float, y1: float, x2: float, y2: float, color: str) -> str:
    """绘制贝塞尔曲线"""
    cx1, cy1 = x1, (y1 + y2) / 2
    cx2, cy2 = x2, (y1 + y2) / 2
    return f'  <path d="M{x1},{y1} C{cx1},{cy1} {cx2},{cy2} {x2},{y2}" stroke="{color}" stroke-width="2" fill="none"/>\n'

def _distribute_nodes(count: int, total_width: int, padding: int, node_width: int) -> List[float]:
    """计算节点均匀分布的x坐标列表"""
    if count == 0:
        return []
    if count == 1:
        return [(total_width - node_width) / 2]
    spacing = (total_width - padding * 2 - count * node_width) / (count - 1)
    return [padding + i * (node_width + spacing) for i in range(count)]

def _generate_error_svg(msg: str) -> str:
    """生成错误提示SVG"""
    return f'''<svg width="500" height="160" xmlns="http://www.w3.org/2000/svg" style="background:{COLORS["light"]};">
  <rect width="100%" height="100%" fill="#FFF3E0" rx="10" stroke="#D0021B" stroke-width="2"/>
  <text x="250" y="65" font-size="18" fill="{COLORS["danger"]}" text-anchor="middle" font-weight="bold">⚠️ SVG生成失败</text>
  <text x="250" y="100" font-size="14" fill="{COLORS["text"]}" text-anchor="middle">{msg}</text>
</svg>'''

