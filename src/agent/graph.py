"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, NotRequired, Required, Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START
from langgraph.runtime import Runtime
from typing_extensions import TypedDict
from pydantic import BaseModel

from langchain.chat_models import init_chat_model

from operator import add

import pandas as pd

import json
import aiohttp  # type: ignore
import re
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

import os

load_dotenv()

SEP = "<SEP>"


def _normalize_text(s: str) -> str:
    s = (s or "").replace("\\7", "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _split_sep(s: str) -> List[str]:
    s = _normalize_text(s)
    if not s:
        return []
    parts = [p.strip() for p in s.split(SEP)]
    return [p for p in parts if p]


def _keywords_from_metadata(meta: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    kw = (meta or {}).get("keywords", {}) or {}
    hi = [_normalize_text(x).lower() for x in kw.get("high_level", []) if x]
    lo = [_normalize_text(x).lower() for x in kw.get("low_level", []) if x]
    return hi, lo


def _score_text(text: str, hi_kw: List[str], lo_kw: List[str]) -> float:
    """
    Simple, robust scoring:
    - +3 for each high-level keyword hit
    - +1 for each low-level keyword hit
    - small bonus for being more 'sentence-like' (length)
    """
    t = _normalize_text(text).lower()
    if not t:
        return 0.0

    score = 0.0
    for k in hi_kw:
        if k and k in t:
            score += 3.0
    for k in lo_kw:
        if k and k in t:
            score += 1.0

    # mild length bonus so we don't pick ultra-short fragments
    score += min(len(t) / 200.0, 0.5)
    return score


def extract_contexts(result: Dict[str, Any], top_n: int = 5) -> List[str]:
    data = result.get("data", {}) or {}
    meta = result.get("metadata", {}) or {}
    hi_kw, lo_kw = _keywords_from_metadata(meta)

    candidates: List[Tuple[float, str]] = []

    # 1) Relationships first: often already "contextual"
    for rel in data.get("relationships", []) or []:
        src = _normalize_text(rel.get("src_id", ""))
        tgt = _normalize_text(rel.get("tgt_id", ""))
        for desc in _split_sep(rel.get("description", "")):
            # Make the sentence explicit + human-readable
            if src and tgt:
                text = f"{src} → {tgt}: {desc}"
            else:
                text = desc
            candidates.append((_score_text(text, hi_kw, lo_kw), text))

    # 2) Entities: convert entity + description into a clean fact
    for ent in data.get("entities", []) or []:
        name = _normalize_text(ent.get("entity_name", ""))
        etype = _normalize_text(ent.get("entity_type", ""))
        for desc in _split_sep(ent.get("description", "")):
            if name:
                # Keep type if it's informative; avoid cluttering with UNKNOWN
                if etype and etype.upper() != "UNKNOWN":
                    text = f"{name} ({etype}): {desc}"
                else:
                    text = f"{name}: {desc}"
            else:
                text = desc
            candidates.append((_score_text(text, hi_kw, lo_kw), text))

    # 3) Deduplicate (preserve best-scoring version)
    best_by_text: Dict[str, float] = {}
    for score, text in candidates:
        if not text:
            continue
        if text not in best_by_text or score > best_by_text[text]:
            best_by_text[text] = score

    ranked = sorted(best_by_text.items(), key=lambda x: x[1], reverse=True)

    # 4) Pick top N, but skip very low-signal lines
    contexts: List[str] = []
    for text, score in ranked:
        if score < 0.2:
            continue
        contexts.append(text)
        if len(contexts) >= top_n:
            break

    return contexts


async def query_knowledge_graph(query: str) -> None:
    url = "http://localhost:9621/query/data"

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }

    payload = {
        "query": query,
        "mode": "mix",
        "only_need_context": True,
        "only_need_prompt": True,
        "response_type": "string",
        "top_k": 20,
        "chunk_top_k": 10,
        "max_entity_tokens": 3500,
        "max_relation_tokens": 2500,
        "max_total_tokens": 6000,
        "enable_rerank": True,
        "include_references": True,
        "include_chunk_content": False,
        "stream": False,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            headers=headers,
            json=payload,  # IMPORTANT: use json= not data=
        ) as response:
            response.raise_for_status()
            result = await response.json()

    final_result = extract_contexts(result)
    return final_result  # type: ignore


class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    rag: bool
    model_name: str


class Case(TypedDict):
    """Defines a single case structure."""

    question: str
    ground_truth: str
    ground_truth_reasoning: str
    contexts: Optional[List[str]]
    answer_reasoning: Optional[str]
    answer: Optional[str]
    source: Optional[str]
    image_paths: Optional[List[str]]


class ModelAnswer(BaseModel):
    """Defines the model's answer structure."""

    answer: Annotated[str, "Diagnosed disease name only."]
    answer_reasoning: Annotated[
        str, "The model's reasoning process leading to the answer."
    ]


class State(TypedDict):
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    input_csv_path: Required[str]
    cases: NotRequired[List[Case]]
    n: NotRequired[int]
    num_cases: NotRequired[int]
    generated_cases: Annotated[NotRequired[List[Case]], add]


def convert_to_base64(image_path: str) -> str:
    """Convert an image file to a base64-encoded string."""
    with open(image_path, "rb") as image_file:
        encoded_string = b64encode(image_file.read()).decode("utf-8")
    return encoded_string


async def call_model(state: State, runtime: Runtime[Context]):
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    # llm = init_chat_model(
    #     model=runtime.context["model_name"],
    #     model_provider="openai",
    #     base_url="http://localhost:8002/v1",
    # )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2,
    )

    current_case = state["cases"].pop()  # type: ignore
    output = current_case.copy()

    llm_structured_output = llm.with_structured_output(ModelAnswer)

    question = current_case["question"]  # type: ignore

    human_msg = f"""Given the following question, provide a concise diagnosis answer and your reasoning.
Question: {question}"""

    if runtime.context.get("rag", False):
        contexts = await query_knowledge_graph(question)
        output["contexts"] = contexts  # type: ignore
        context_str = "\n\n".join(contexts)  # type: ignore
        human_msg += f"\n\nRelevant Contexts:\n{context_str}"

    human_content = [
        {"type": "text", "text": human_msg},
    ]

    image_lists = [
        {"type": "image", "base64": convert_to_base64(img_path), "mime_type": "image/jpeg"} for img_path in current_case.get("image_paths", [])  # type: ignore
    ]

    if image_lists:
        human_content.extend(image_lists)

    msgs = [
        {
            "role": "system",
            "content": "You are a medical diagnosis assistant which have a deep understanding of tropical diseases.",
        },
        {
            "role": "user",
            "content": human_content,
        },
    ]
    response: ModelAnswer = await llm_structured_output.ainvoke(msgs)  # type: ignore

    output["answer"] = response.answer
    output["answer_reasoning"] = response.answer_reasoning

    return {"generated_cases": [output], "n": state["n"] - 1}  # type: ignore


def query_context(question: str) -> List[str]:
    """Mock function to retrieve relevant contexts for a given question."""
    # In a real implementation, this would query a knowledge base or use embeddings.
    return [
        "Esophageal carcinoma commonly presents with progressive dysphagia that advances from solids to liquids and is frequently associated with weight loss.",
        "Benign strictures are usually related to chronic reflux or caustic injury and are less commonly associated with marked weight loss.",
    ]


def extract_cases_from_csv(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Extract cases from CSV file."""

    df = pd.read_csv(state["input_csv_path"])
    cases_raw = df.to_dict(orient="records")
    cases: List[Case] = [
        parse_case_input(cr) for cr in cases_raw if cr.get("flagged") == "No"
    ]

    return {"cases": cases, "num_cases": len(cases), "n": len(cases)}


def parse_case_input(case_input: Dict) -> Case:
    """Parse a single input dict into a `Case` TypedDict.

    Fields `contexts`, `answer_reasoning` and `answer` may be None.
    """
    imgs_folder = "/Users/fuisloy/medgenerate/benchmark_image"
    return Case(
        question=case_input.get("case_prompt") or case_input.get("question") or "",
        ground_truth=case_input.get("final_diagnosis") or "",
        ground_truth_reasoning=case_input.get("reasoning_narrative") or "",
        contexts=None,
        answer_reasoning=None,
        answer=None,
        source=case_input.get("source") or None,
        image_paths=(
            [
                os.path.join(imgs_folder, case_input.get("source", ""), img)
                for img in os.listdir(
                    os.path.join(imgs_folder, case_input.get("source", ""))
                )
                if os.path.isfile(
                    os.path.join(imgs_folder, case_input.get("source", ""), img)
                )
            ]
            if case_input.get("source")
            and os.path.isdir(os.path.join(imgs_folder, case_input.get("source", "")))
            else None
        ),
    )


def decide_next_step(state: State, runtime: Runtime[Context]):
    """Decide the next step based on remaining cases."""
    if state.get("n", 0) > 0:
        return "call_model"
    else:
        return "export_results"


def export_results(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Export generated cases to JSON file."""
    model_name = runtime.context["model_name"]
    rag_enabled = runtime.context.get("rag", False)
    output_path = (
        model_name.replace("/", "_")
        + "_generated_cases"
        + ("_rag" if rag_enabled else "")
        + ".json"
    )
    with open(output_path, "w") as f:

        json.dump(
            {
                "model_name": model_name,
                "rag": rag_enabled,
                "test_cases": state.get("generated_cases", []),
            },
            f,
            indent=4,
        )
    return {"output_path": output_path}


"""Example json output
{
    "test_cases": [
        {
            "question": "A 62-year-old man presented with a 4-month history of progressive dysphagia that initially involved solid foods and later progressed to difficulty swallowing liquids. He also reported unintentional weight loss of approximately 8 kg over this period. He denied odynophagia, hematemesis, or prior history of gastroesophageal reflux disease. Physical examination was unremarkable. Laboratory studies showed mild normocytic anemia. Contrast-enhanced CT of the chest demonstrated an irregular, circumferential thickening of the mid-esophagus with luminal narrowing. A barium swallow study showed an irregular narrowing with shouldered edges. There was no evidence of distant metastasis on imaging.",
            "ground_truth": "esophageal malignancy",
            "ground_truth_reasoning": "Progressive dysphagia suggests esophageal malignancy.\nWeight loss supports a malignant process.",
            "contexts": [
                "Esophageal carcinoma commonly presents with progressive dysphagia that advances from solids to liquids and is frequently associated with weight loss.",
                "Benign strictures are usually related to chronic reflux or caustic injury and are less commonly associated with marked weight loss."
            ],
            "answer_reasoning": "Significant weight loss further supports a malignant etiology.",
            "answer": "progressive dysphagia"
        }
    ]
}
"""


builder = StateGraph(State, context_schema=Context)

builder.add_node("extract_cases_from_csv", extract_cases_from_csv)
builder.add_node("call_model", call_model)
builder.add_node("export_results", export_results)

builder.add_edge(START, "extract_cases_from_csv")
builder.add_edge("extract_cases_from_csv", "call_model")
builder.add_conditional_edges(
    "call_model",
    decide_next_step,
    {
        "call_model": "call_model",
        "export_results": "export_results",
    },
)  # type: ignore
builder.add_edge("export_results", END)

graph = builder.compile()
