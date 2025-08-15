#!/usr/bin/env python3
"""
AI Tutor / Learning Coach (Functional Demo)

Demonstrates a working tutor that:
- Runs short quizzes by topic
- Stores quiz results as EPISODIC memories with event_date and topics
- Tracks weak concepts as SEMANTIC memories
- Suggests what to practice next based on recent performance
- Provides a recent summary

Two modes:
- Interactive (default): REPL commands
- Demo (--demo): runs a mini sequence across topics and shows suggestions/summary

Environment variables:
- MEMORY_SERVER_URL (default: http://localhost:8000)
"""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from agent_memory_client import MemoryAPIClient, create_memory_client
from agent_memory_client.filters import CreatedAt, MemoryType, Namespace, Topics
from agent_memory_client.models import ClientMemoryRecord, MemoryTypeEnum
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


DEFAULT_USER = "student"
DEFAULT_SESSION = "tutor_session"
MEMORY_SERVER_URL = os.getenv("MEMORY_SERVER_URL", "http://localhost:8000")


def _namespace(user_id: str) -> str:
    return f"ai_tutor:{user_id}"


async def _get_client() -> MemoryAPIClient:
    return await create_memory_client(base_url=MEMORY_SERVER_URL, timeout=30.0)


def _get_llm() -> ChatOpenAI | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    return ChatOpenAI(model="gpt-4o", temperature=0)


GENERATE_QUESTIONS_FN = {
    "name": "generate_quiz",
    "description": "Generate a short quiz for a topic.",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "answer": {"type": "string"},
                        "concept": {"type": "string"},
                    },
                    "required": ["prompt", "answer", "concept"],
                },
            }
        },
        "required": ["questions"],
    },
}


GRADE_ANSWER_FN = {
    "name": "grade_answer",
    "description": "Grade a student's answer and provide a brief feedback.",
    "parameters": {
        "type": "object",
        "properties": {
            "correct": {"type": "boolean"},
            "feedback": {"type": "string"},
            "concept": {"type": "string"},
        },
        "required": ["correct", "feedback"],
    },
}


def _llm_bind(*functions: dict) -> ChatOpenAI | None:
    llm = _get_llm()
    if not llm:
        return None
    return llm.bind_functions(list(functions))


@dataclass
class Question:
    prompt: str
    answer: str
    concept: str


QUIZZES: dict[str, list[Question]] = {
    "algebra": [
        Question("Solve: 2x + 3 = 9. x = ?", "3", "linear_equations"),
        Question("What is the slope in y = 5x + 1?", "5", "slope"),
    ],
    "geometry": [
        Question("Sum of interior angles in a triangle?", "180", "triangles"),
        Question("Area of a circle with r=3?", "28.27", "circle_area"),
    ],
}


async def record_quiz_result(
    user_id: str, topic: str, concept: str, correct: bool
) -> None:
    client = await _get_client()
    ns = _namespace(user_id)
    # Episodic memory: per-question result
    epi = ClientMemoryRecord(
        text=f"Quiz result: topic={topic}, concept={concept}, correct={correct}",
        memory_type=MemoryTypeEnum.EPISODIC,
        topics=["quiz", topic, concept],
        namespace=ns,
        user_id=user_id,
        event_date=datetime.now(UTC),
    )
    await client.create_long_term_memory([epi])

    # Semantic memory: update weak concepts when incorrect
    if not correct:
        weak = ClientMemoryRecord(
            text=f"Weak concept: {concept} in {topic}",
            memory_type=MemoryTypeEnum.SEMANTIC,
            topics=["weak_concept", topic, concept],
            namespace=ns,
            user_id=user_id,
        )
        await client.create_long_term_memory([weak])


async def get_weak_concepts(user_id: str, since_days: int = 30) -> list[str]:
    client = await _get_client()
    ns = _namespace(user_id)
    results = await client.search_long_term_memory(
        text="weak concepts",
        namespace=Namespace(eq=ns),
        topics=Topics(any=["weak_concept"]),
        memory_type=MemoryType(eq="semantic"),
        created_at=CreatedAt(gte=(datetime.now(UTC) - timedelta(days=since_days))),
        limit=50,
        optimize_query=False,
    )
    concepts: list[str] = []
    for m in results.memories:
        # text format: "Weak concept: {concept} in {topic}"
        text = m.text
        if text.startswith("Weak concept: "):
            payload = text[len("Weak concept: ") :]
            concepts.append(payload)
    return concepts


async def practice_next(user_id: str) -> str:
    concepts = await get_weak_concepts(user_id, since_days=30)
    if not concepts:
        return "You're doing great! No weak concepts detected recently."
    return f"Focus next on: {', '.join(concepts[:3])}"


async def recent_summary(user_id: str, since_days: int = 7) -> list[str]:
    client = await _get_client()
    ns = _namespace(user_id)
    results = await client.search_long_term_memory(
        text="recent quiz",
        namespace=Namespace(eq=ns),
        topics=Topics(any=["quiz"]),
        memory_type=MemoryType(eq="episodic"),
        created_at=CreatedAt(gte=(datetime.now(UTC) - timedelta(days=since_days))),
        limit=100,
        optimize_query=False,
    )
    return [m.text for m in results.memories]


async def run_quiz(user_id: str, topic: str) -> None:
    questions = QUIZZES.get(topic)
    llm = _llm_bind(GENERATE_QUESTIONS_FN, GRADE_ANSWER_FN)
    if llm and not questions:
        # Ask the LLM to generate a small quiz
        system = {
            "role": "system",
            "content": "Generate 2 concise questions via function call.",
        }
        user = {"role": "user", "content": f"Create a quiz for topic: {topic}."}
        resp = llm.invoke([system, user])
        fn = getattr(resp, "additional_kwargs", {}).get("function_call")
        if fn and fn.get("name") == "generate_quiz":
            import json as _json

            try:
                args = (
                    _json.loads(fn["arguments"])
                    if isinstance(fn.get("arguments"), str)
                    else fn.get("arguments", {})
                )
            except Exception:
                args = {}
            qs = args.get("questions", [])
            questions = [
                Question(
                    prompt=q.get("prompt", ""),
                    answer=q.get("answer", ""),
                    concept=q.get("concept", topic),
                )
                for q in qs
            ]

    if not questions:
        print("Unknown topic")
        return
    correct_count = 0
    total = len(questions)
    for q in questions:
        print(q.prompt)
        ans = input("Your answer: ").strip()
        correct = _normalize(ans) == _normalize(q.answer)
        graded_feedback = None
        if llm:
            # Let LLM grade and provide feedback
            system = {
                "role": "system",
                "content": "Grade and respond via function call only.",
            }
            user = {
                "role": "user",
                "content": f"Question: {q.prompt}\nExpected: {q.answer}\nStudent: {ans}",
            }
            resp = llm.invoke([system, user])
            fn = getattr(resp, "additional_kwargs", {}).get("function_call")
            if fn and fn.get("name") == "grade_answer":
                import json as _json

                try:
                    args = (
                        _json.loads(fn["arguments"])
                        if isinstance(fn.get("arguments"), str)
                        else fn.get("arguments", {})
                    )
                except Exception:
                    args = {}
                graded_feedback = args.get("feedback")
                correct = bool(args.get("correct", correct))
        print("Correct!" if correct else f"Incorrect. Expected {q.answer}")
        if graded_feedback:
            print(f"Feedback: {graded_feedback}")
        await record_quiz_result(user_id, topic, q.concept, correct)
        if correct:
            correct_count += 1
    print(f"Score: {correct_count}/{total}")


def _normalize(s: str) -> str:
    return s.strip().lower()


async def run_demo(user_id: str, session_id: str) -> None:
    print("🎓 AI Tutor Demo")
    # Simulate a short run with preset answers
    demo_answers = {
        ("algebra", 0): "3",  # correct
        ("algebra", 1): "4",  # incorrect (slope 5)
        ("geometry", 0): "180",  # correct
        ("geometry", 1): "28.27",  # correct
    }
    for topic in ("algebra", "geometry"):
        for i, q in enumerate(QUIZZES[topic]):
            ans = demo_answers.get((topic, i), "")
            correct = _normalize(ans) == _normalize(q.answer)
            await record_quiz_result(user_id, topic, q.concept, correct)
            print(
                f"{topic}: {q.prompt} -> {ans} ({'correct' if correct else 'incorrect'})"
            )

    print("\nWeak concepts:")
    for c in await get_weak_concepts(user_id):
        print(f"- {c}")

    print("\nPractice next:")
    print(await practice_next(user_id))

    print("\nRecent summary:")
    for line in await recent_summary(user_id):
        print(f"- {line}")


async def run_interactive(user_id: str, session_id: str) -> None:
    print("🎓 AI Tutor - Interactive Mode")
    print(
        "Commands:\n  quiz <topic> (options: algebra, geometry)\n  practice-next\n  weak-concepts\n  summary [--days N]\n  exit"
    )
    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye")
            return
        if not raw:
            continue
        if raw.lower() in {"exit", "quit"}:
            print("Bye")
            return

        parts = raw.split()
        cmd = parts[0]
        try:
            if cmd == "quiz" and len(parts) > 1:
                await run_quiz(user_id, parts[1])
            elif cmd == "practice-next":
                print(await practice_next(user_id))
            elif cmd == "weak-concepts":
                for c in await get_weak_concepts(user_id):
                    print(f"- {c}")
            elif cmd == "summary":
                days = 7
                if "--days" in parts:
                    i = parts.index("--days")
                    if i + 1 < len(parts):
                        days = int(parts[i + 1])
                for line in await recent_summary(user_id, days):
                    print(f"- {line}")
            else:
                print("Unknown command")
        except Exception as e:  # noqa: BLE001
            print(f"Error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Tutor")
    parser.add_argument("--user-id", default=DEFAULT_USER)
    parser.add_argument("--session-id", default=DEFAULT_SESSION)
    parser.add_argument("--memory-server-url", default=MEMORY_SERVER_URL)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.memory_server_url:
        os.environ["MEMORY_SERVER_URL"] = args.memory_server_url

    if args.demo:
        asyncio.run(run_demo(args.user_id, args.session_id))
    else:
        asyncio.run(run_interactive(args.user_id, args.session_id))


if __name__ == "__main__":
    main()
