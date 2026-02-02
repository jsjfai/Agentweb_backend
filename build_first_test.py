import asyncio
import json
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from config_loader import load_config
from openai import OpenAI

config = load_config()

# === 读大模型配置 ===
def llm_config():
    return OpenAI(
        api_key=config["LLMConfigure_qwen3"]["api_key"],
        base_url=config["LLMConfigure_qwen3"]["base_url"],
    )

# === 使用 llm 生成 arguments ===
def generate_args_by_llm(tool_schema, user_query: str) -> dict:
    """
    使用 llm 根据 inputSchema + 用户查询生成 arguments。
    自动处理 district_id，返回城市级的6位adcode（例如北京=110100，而不是110000）。
    """
    client = llm_config()

    system_prompt = (
        "You are a JSON argument generator. "
        "Given a tool inputSchema (JSON Schema) and a user's natural-language query, "
        "produce a minimal valid JSON object that satisfies the schema. "
        "⚠️ Important rules:\n"
        "1. ONLY output the JSON object, nothing else.\n"
        "2. If the schema includes 'district_id', return the correct 6-digit adcode "
        "for the mentioned city at the CITY level (not province). Example: "
        "'北京' → '110100', '上海' → '310100'.\n"
        "3. Ensure the result is valid JSON, no extra commentary."
    )

    resp = client.chat.completions.create(
        model=config["LLMConfigure_qwen3"]["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Schema: {tool_schema}\nQuery: {user_query}"},
        ],
        temperature=0,
    )

    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"error": f"Invalid JSON from model: {raw}"}

# === 使用llm流式解释结果 ===
async def explain_result_stream(smart_result: dict):
    """
    用 DeepSeek 把 smart_result 转成中文解释，流式输出
    """
    client = llm_config()
    stream = client.chat.completions.create(
        model=config["LLMConfigure_qwen3"]["model"],
        messages=[
            {"role": "system", "content": "你是一个结果解释助手。请将提供的 JSON 查询结果转换为简明扼要的中文说明，便于普通用户理解。"},
            {"role": "user", "content": json.dumps(smart_result, ensure_ascii=False)},
        ],
        stream=True,
    )
    # print("\n=== 服务执行结果（中文） ===")
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print("\n")  # 换行

# === 核心逻辑 ===
async def smart_query(baseurl: str, user_query: str, debug: bool = False):
    # print(f"Connecting to: {baseurl}")
    async with streamablehttp_client(baseurl) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Step 1: get_available_tools
            print("请求 get_available_tools ...")
            tools_result = await session.call_tool("get_available_tools", {"query": user_query})
            if tools_result.content and hasattr(tools_result.content[0], "text"):
                try:
                    tools_data = json.loads(tools_result.content[0].text)
                except Exception as e:
                    print("解析工具列表失败:", e)
                    return None
            else:
                print("未返回工具列表")
                return None

            tools = tools_data.get("tools", [])
            print(f"get_available_tools -> {len(tools)} candidate tools (metadata={tools_data.get('metadata')})")

            # Step 2: 遍历工具，尝试执行
            attempts = []
            final_result = None
            for tool in tools:
                tool_name = tool.get("name")
                schema = tool.get("inputSchema", {})

                print(f"\n尝试工具: {tool_name}")
                args = generate_args_by_llm(schema, user_query)
                print("  生成的 arguments:", args)

                try:
                    exec_result = await session.call_tool("execute_tool", {
                        "toolName": tool_name,
                        "arguments": args
                    })

                    # 解析结果
                    if exec_result.content and hasattr(exec_result.content[0], "text"):
                        text_result = exec_result.content[0].text
                        try:
                            parsed_result = json.loads(text_result)
                        except Exception:
                            parsed_result = text_result
                    else:
                        parsed_result = exec_result.structuredContent or None

                    # print("  执行结果解析：", parsed_result)

                    attempts.append({"tool": tool_name, "args": args, "result": parsed_result})

                    # 先找到第一个能返回结果的工具就停止
                    if parsed_result and not isinstance(parsed_result, str):
                        final_result = parsed_result
                        break
                except Exception as e:
                    print(f"  执行工具 {tool_name} 出错:", e)

            smart_result = {
                "found": final_result is not None,
                "tool": tool_name if final_result else None,
                "arguments": args if final_result else None,
                "result": final_result if final_result else None,
                "attempts": attempts,
            }

            # === 输出逻辑 ===
            await explain_result_stream(smart_result)

            # if debug:
            #     print("\n=== Smart 服务执行结果（JSON 调试信息） ===")
            #     print(json.dumps(smart_result, ensure_ascii=False, indent=2))

            return smart_result

# === 测试入口 ===
if __name__ == "__main__":
    baseurl = "http://192.168.201.180:30013/mcp/$smart"
    # query = "查询北京的天气信息"
    query = "高级服装定制"

    result = asyncio.run(smart_query(baseurl, query, debug=True))
