from __future__ import annotations

from langchain_core.messages import AIMessage

from agents import (
    build_graph,
    create_runtime_cache,
    merge_runtime_cache_into_state,
    update_runtime_cache_from_state,
)
from config import DEFAULT_MODEL_CONFIG
from memory import get_checkpointer
from models import create_llm

async def ask_with_hybrid_memory(
    user_prompt: str,
    thread_id: str,
    runtime_cache: dict | None = None,
) -> tuple[str, dict]:
    if runtime_cache is None:
        runtime_cache = create_runtime_cache()

    llm = create_llm(DEFAULT_MODEL_CONFIG)
    initial_state = merge_runtime_cache_into_state(user_prompt, runtime_cache)

    async with get_checkpointer() as checkpointer:
        graph = build_graph(llm).compile(checkpointer=checkpointer)

        result = await graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )

    update_runtime_cache_from_state(runtime_cache, result)

    final_answer = "No relevant match found."
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            if msg.content:
                final_answer = msg.content
                break

    return final_answer, runtime_cache