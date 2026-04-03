from __future__ import annotations

import re
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from tools import call_tool, get_available_tools


SYSTEM_PROMPT = """
You are a master database engineer with expertise in SQLite query construction and optimisation.
Your purpose is to transform natural language requests into precise, efficient SQL queries that deliver exactly what the user needs.

Your job:
1. Implement your own strategic plan to explore and understand the database with multiple tables before constructing the sql query.
2. Determine the most efficient way to investigate all the tables in the database based on the user's request.
3. Extract the key meaning from the user's request and contextualize it with the table structure. The user may not use exact table or column wording.
4. Independently identify which database, table, and column names need to be examined.
5. Formulate the query based on real table structure only.
6. Explain your reasoning clearly.
7. Do not hallucinate or invent data.
8. Do not assume answers come from one table only; joins may be required.

Your responses should: 
- be formatted as Markdown.
- Return final values in a readable format, not raw SQL tables.
Your target audience may not be familiar with SQL.
""".strip()


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    rules: list[str]
    table_info_cache: dict[str, str]
    table_columns: dict[str, list[str]]
    table_samples: dict[str, str]
    last_user_prompt: str


def create_runtime_cache() -> dict[str, Any]:
    """
    Fast in-session cache for the current browser/app session.
    This avoids repeated DB/thread reads while the session is still active.
    """
    return {
        "rules": [],
        "table_info_cache": {},
        "table_columns": {},
        "table_samples": {},
    }


def parse_columns_from_table_info(table_info: str) -> list[str]:
    m = re.search(r"CREATE TABLE .*?\((.*?)\)\s*(/\*|$)", table_info, flags=re.S | re.I)
    if not m:
        return []

    body = m.group(1)
    cols: list[str] = []

    for line in body.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue

        upper = line.upper()
        if upper.startswith(("PRIMARY KEY", "FOREIGN KEY", "UNIQUE", "CONSTRAINT", "KEY", "INDEX")):
            continue

        parts = line.split()
        if parts:
            cols.append(parts[0].strip('`"'))

    return cols


def extract_rules_from_prompt(prompt: str, existing_rules: list[str]) -> list[str]:
    rules = list(existing_rules)
    lower = prompt.lower()

    candidates: list[str] = []

    if "no sql" in lower or "only readable text" in lower:
        candidates.append("User wants readable text output and no SQL in the response.")

    if "simple explanation" in lower or "explain simply" in lower:
        candidates.append("User prefers simple explanations.")

    for rule in candidates:
        if rule not in rules:
            rules.append(rule)

    return rules


def build_schema_memory_message(state: AgentState) -> str:
    if not state["table_info_cache"]:
        return "No tables have been inspected yet."

    parts = ["Previously inspected tables:"]
    for table_name, table_info in state["table_info_cache"].items():
        parts.append(f"\n### {table_name}\n{table_info}")
    return "\n".join(parts)


def build_rules_memory_message(state: AgentState) -> str:
    if not state["rules"]:
        return "No session rules recorded yet."

    parts = ["User/session rules:"]
    for rule in state["rules"]:
        parts.append(f"- {rule}")
    return "\n".join(parts)


def merge_runtime_cache_into_state(
    user_prompt: str,
    runtime_cache: dict[str, Any],
) -> AgentState:
    """
    Seed the graph state from the in-session cache.
    This gives low latency during the active session.
    """
    return {
        "messages": [HumanMessage(content=user_prompt)],
        "rules": list(runtime_cache.get("rules", [])),
        "table_info_cache": dict(runtime_cache.get("table_info_cache", {})),
        "table_columns": dict(runtime_cache.get("table_columns", {})),
        "table_samples": dict(runtime_cache.get("table_samples", {})),
        "last_user_prompt": user_prompt,
    }


def update_runtime_cache_from_state(
    runtime_cache: dict[str, Any],
    state: AgentState,
) -> None:
    """
    Push latest graph state back into the fast local cache.
    """
    runtime_cache["rules"] = list(state.get("rules", []))
    runtime_cache["table_info_cache"] = dict(state.get("table_info_cache", {}))
    runtime_cache["table_columns"] = dict(state.get("table_columns", {}))
    runtime_cache["table_samples"] = dict(state.get("table_samples", {}))


def agent_node(state: AgentState, llm: BaseChatModel) -> dict[str, Any]:
    messages = state["messages"]

    last_user_prompt = state.get("last_user_prompt", "")
    if not last_user_prompt:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_prompt = msg.content
                break

    updated_rules = extract_rules_from_prompt(last_user_prompt, state["rules"])
    tmp_state: AgentState = {
        **state,
        "rules": updated_rules,
    }

    schema_memory = build_schema_memory_message(tmp_state)
    rules_memory = build_rules_memory_message(tmp_state)

    llm_with_tools = llm.bind_tools(get_available_tools()) # Tool-bound model 
    ai_msg = llm_with_tools.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            *messages,
            HumanMessage(
                content=(
                    "Use tools whenever needed to inspect schema and query the database.\n"
                    "Do not assume table names or columns.\n"
                    "Do not return SQL as the final answer.\n\n"
                    f"{schema_memory}\n\n"
                    "Also follow these session rules/preferences:\n\n"
                    f"{rules_memory}"
                )
            ),
        ]
    )

    return {
        "messages": [ai_msg],
        "rules": updated_rules,
        "last_user_prompt": last_user_prompt,
    }


def tools_node(state: AgentState) -> dict[str, Any]:
    last_ai = state["messages"][-1]
    updated_table_info_cache = dict(state["table_info_cache"])
    updated_table_columns = dict(state["table_columns"])
    updated_table_samples = dict(state["table_samples"])
    out_messages: list[AnyMessage] = []

    for tool_call in getattr(last_ai, "tool_calls", []):
        tool_msg = call_tool(tool_call)
        out_messages.append(tool_msg)

        if tool_call["name"] == "describe_table":
            table_name = tool_call["args"]["table_name"]
            table_info = str(tool_msg.content)

            updated_table_info_cache[table_name] = table_info
            updated_table_columns[table_name] = parse_columns_from_table_info(table_info)

            sample_match = re.search(
                rf"\d+\s+rows from\s+{re.escape(table_name)}\s+table:\n(.*)",
                table_info,
                flags=re.S | re.I,
            )
            updated_table_samples[table_name] = sample_match.group(1).strip() if sample_match else ""

    return {
        "messages": out_messages,
        "table_info_cache": updated_table_info_cache,
        "table_columns": updated_table_columns,
        "table_samples": updated_table_samples,
    }


def should_continue(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "tools"
    return END


def build_graph(llm: BaseChatModel):
    builder = StateGraph(AgentState)

    builder.add_node("agent", lambda state: agent_node(state, llm))
    builder.add_node("tools", tools_node)

    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )
    builder.add_edge("tools", "agent")

    return builder