# 基于 LangGraph 的多智能体协作系统

基于 LangGraph StateGraph 与 Qwen 大模型，实现自然语言到 SVG 信息图的自动转换，支持流程图、时间线、对比柱状图、层次结构图、概念关系图 5 种图表类型。

## 运行说明

1. 本项目调用外部 LLM API，需在代码中配置 API Key。
2. `data/示例.csv` 展示了输入数据格式，将数据填入即可自动获取生成内容。
3. 在 `main.py` 中填入需要生成的 ID。
4. 安装依赖后运行：

```bash
pip install -r requirements.txt
python main.py
