from __future__ import annotations

from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from log_utils import log, log_panel, green_border_style
from tools import call_tool, get_available_tools


SYSTEM_PROMPT = """
You are a master database engineer with expertise in SQLite query construction and optimisation. 
Your purpose is to transform natural language requests into precise, efficient SQL queries that deliver exactly what the user needs. 


Your job:
1. Implement your own strategic plan to explore and understand the database with multiple tables before constructing the sql query. 
2. Determine the most efficient way to investigating through the all the tables in the database based on user's requests. 
3. Only extract the key words from users' requests, understand users' requirement and contextualize it with the table structure. ie. might not be word for word. 
4. Independently identify which database/ table/ column names need to be examined to fulfill the query requirements. 
5. Formulate the query based on your understanding of the database table structure. 
6. For every decision made, need to provide me with detailed explanation. 
7. Do not hallucinate or make up data that does not exist in the database. 

Your responses should be formatted as Markdown. Prefer using tables, lists or graphs for displaying data where appropriate. 
Your target audience is people who may not be familiar with SQL.   
""".strip()


def create_history() -> List[BaseMessage]:
    return [SystemMessage(content=SYSTEM_PROMPT)]

def ask(
    prompt: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    max_iterations: int = 10,
    callback=None,
) -> str:
    log_panel(title="User Request", content=f"Query: {prompt}", border_style=green_border_style)

    n_iteration = 0
    messages = history.copy()
    messages.append(HumanMessage(content=prompt))

    tools_by_name = {tool.name: tool for tool in get_available_tools()}

    while n_iteration < max_iterations:
        ai_msg = llm.invoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            return ai_msg.content

        for tool_call in ai_msg.tool_calls:
            tool_msg = call_tool(tool_call, callback=callback)
            messages.append(tool_msg)

            # auto-describe all tables after list_tables
            if tool_call["name"] == "list_tables":
                try:
                    rows = tools_by_name["list_tables"].invoke({})
                    tables = [row[0] for row in rows if row]
                except Exception:
                    tables = []

                for table_name in tables:
                    describe_call = {
                        "id": f"{tool_call['id']}_{table_name}",
                        "name": "describe_table",
                        "args": {
                            "table_name": table_name,
                            "reasoning": f"Auto-inspect schema for {table_name}"
                        },
                    }
                    describe_msg = call_tool(describe_call, callback=callback)
                    messages.append(describe_msg)

        n_iteration += 1

    raise RuntimeError(
        "Maximum number of iterations reached. Please try again with a different query."
    )