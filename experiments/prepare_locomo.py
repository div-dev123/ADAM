"""Converts official Snap-Research / Stanford LoCoMo multi-session data into ADAM benchmark format."""

import json
from pathlib import Path
import re
from typing import List


def prepare_locomo_benchmark(
    source_json_path: str = "experiments/data/locomo10.json",
    output_json_path: str = "experiments/data/locomo_benchmark.json",
    sample_index: int = 0,
    num_sessions: int = None,
):
    source_path = Path(source_json_path)
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sample = data[sample_index]
    conv = sample["conversation"]
    qas = sample["qa"]
    speaker_a = conv.get("speaker_a", "Caroline")
    speaker_b = conv.get("speaker_b", "Melanie")

    # Discover all session keys in numerical order
    all_sess_keys = [
        k for k in conv.keys()
        if k.startswith("session_") and not k.endswith("_date_time")
    ]
    all_sess_keys.sort(key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else 999)
    if num_sessions is not None:
        all_sess_keys = all_sess_keys[:num_sessions]

    turns = []
    turn_id = 1
    dia_ids_present = set()

    for s_idx, s_key in enumerate(all_sess_keys, start=1):
        raw_turns = conv.get(s_key, [])

        for t in raw_turns:
            dia_id = t.get("dia_id", f"D{s_idx}:{turn_id}")
            dia_ids_present.add(dia_id)
            speaker = t.get("speaker", speaker_a)
            text = t.get("text", "").strip()

            role = "user" if speaker == speaker_a else "assistant"

            # Determine turn category
            words = text.lower().split()
            if len(words) <= 6 and any(w in words for w in ["hey", "hello", "hi", "good", "cool", "yeah", "thanks"]):
                category = "filler"
            else:
                category = "fact"

            turns.append({
                "turn_id": turn_id,
                "role": role,
                "content": f"[{speaker}]: {text}",
                "category": category,
                "fact_id": dia_id,
                "tags": [f"session_{s_idx}", speaker.lower()],
            })
            turn_id += 1

    # Inject realistic temporal evolution & contradictions across later sessions
    # e.g., Caroline moves to Seattle (session 8), and updates current primary counseling focus
    contradiction_updates = [
        {
            "role": "user",
            "content": f"[{speaker_a}]: Update on my life: I just settled into my new home in Seattle, Washington! Loving the Pacific Northwest.",
            "category": "contradiction",
            "fact_id": "CONTRA:1",
            "tags": ["session_8", speaker_a.lower(), "location_update"],
        },
        {
            "role": "user",
            "content": f"[{speaker_a}]: As for my career direction, I decided to focus exclusively on adolescent youth counseling as my specialty.",
            "category": "contradiction",
            "fact_id": "CONTRA:2",
            "tags": ["session_12", speaker_a.lower(), "career_update"],
        },
    ]
    for update in contradiction_updates:
        turns.append({
            "turn_id": turn_id,
            "role": update["role"],
            "content": update["content"],
            "category": update["category"],
            "fact_id": update["fact_id"],
            "tags": update["tags"],
        })
        turn_id += 1

    # Extract matching probe queries from LoCoMo QA pairs
    queries = []
    q_counter = 1
    for qa_item in qas:
        ev_list = qa_item.get("evidence", [])
        if not ev_list or not all(e in dia_ids_present for e in ev_list):
            continue

        q_text = qa_item.get("question", "").strip()
        ans_raw = str(qa_item.get("answer", "")).strip()

        # Clean target facts for substring matching in evaluation
        target_facts = [ans_raw]
        # Also add significant comma/semicolon chunks if multi-part
        parts = [p.strip() for p in re.split(r"[,;]", ans_raw) if len(p.strip()) >= 3]
        for p in parts:
            if p not in target_facts:
                target_facts.append(p)

        queries.append({
            "query_id": f"locomo_q{q_counter:02d}",
            "query": q_text,
            "target_facts": target_facts,
            "obsolete_facts": [],
            "category": "locomo_qa",
        })
        q_counter += 1

    # Add targeted Contradiction Evaluation Probes
    queries.append({
        "query_id": f"locomo_q{q_counter:02d}",
        "query": "Where does Caroline live now after relocating?",
        "target_facts": ["Seattle, Washington", "Seattle"],
        "obsolete_facts": ["Sweden"],
        "category": "contradiction_probe",
    })
    q_counter += 1
    queries.append({
        "query_id": f"locomo_q{q_counter:02d}",
        "query": "What is Caroline's specific focus in counseling now?",
        "target_facts": ["adolescent youth counseling", "adolescent youth"],
        "obsolete_facts": ["general adult counseling"],
        "category": "contradiction_probe",
    })
    q_counter += 1

    benchmark_dict = {
        "name": f"LoCoMo_Sample_{sample_index+1}_MultiSession",
        "description": f"Real-world multi-session benchmark from Snap-Research/Stanford LoCoMo dataset ({len(turns)} turns across {num_sessions} sessions with {len(queries)} verified QA probes).",
        "user_id": f"locomo_{speaker_a.lower()}",
        "turns": turns,
        "queries": queries,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_dict, f, indent=2)

    print(f"Successfully generated LoCoMo benchmark at {output_path}")
    print(f"Total turns: {len(turns)}, Total queries: {len(queries)}")
    return benchmark_dict


if __name__ == "__main__":
    prepare_locomo_benchmark()
