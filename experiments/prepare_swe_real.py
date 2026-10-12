"""Converts authentic software engineering trajectories from SWE-agent into ADAM benchmark format."""

import json
from pathlib import Path
import pandas as pd
from typing import Any, Dict, List


def prepare_swe_real_benchmark(
    source_parquet_path: str = "experiments/data/swe_nebius.parquet",
    output_json_path: str = "experiments/data/swe_real_benchmark.json",
) -> Dict[str, Any]:
    parquet_path = Path(source_parquet_path)
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)

    # We select two authentic high-quality solved trajectories:
    # 1. ithaka__apiron-8 (row 3989): Microservices HTTP client, ServiceCaller, authentication argument pass-through
    # 2. astarte-platform__astarte-device-sdk-python-98 (row 1591): IoT cloud device SDK, empty BSON payload handling
    row_apiron = df.iloc[3989]
    row_astarte = df.iloc[1591]

    turns: List[Dict[str, Any]] = []
    turn_id = 1

    # -------------------------------------------------------------------------
    # Session 1: apiron HTTP Client & Authentication (ithaka__apiron-8)
    # -------------------------------------------------------------------------
    traj_apiron = row_apiron["trajectory"]
    for i, step in enumerate(traj_apiron):
        raw_role = step.get("role", "user")
        if raw_role == "system":
            # Convert system instruction to initial developer specification
            role = "user"
            content = f"[Lead Architect]: Starting development sprint on ithaka/apiron repository. Issue #8: Add pass-through auth argument to ServiceCaller.call method so that callers can supply custom authentication credentials."
            cat = "fact"
            fact_id = "apiron_goal"
        elif raw_role == "ai":
            role = "assistant"
            content = f"[Coding Agent]: {str(step.get('text', '')).strip()}"
            cat = "fact" if any(w in content for w in ["ServiceCaller", "auth", "client.py", "def call"]) else "filler"
            fact_id = f"apiron_ai_{i}"
        else:
            role = "user"
            text = str(step.get("text", "")).strip()
            content = f"[Developer / Terminal]: {text}"
            words = text.lower().split()
            if len(words) <= 5 and any(w in words for w in ["ok", "done", "next", "bash-$"]):
                cat = "filler"
            else:
                cat = "fact"
            fact_id = f"apiron_dev_{i}"

        turns.append({
            "turn_id": turn_id,
            "role": role,
            "content": content,
            "category": cat,
            "fact_id": fact_id,
            "tags": ["session_apiron", "http_client", "auth", "swe_real"],
        })
        turn_id += 1

    # -------------------------------------------------------------------------
    # Session 2: Astarte IoT Device Telemetry SDK (astarte-device-sdk-python-98)
    # -------------------------------------------------------------------------
    traj_astarte = row_astarte["trajectory"]
    for i, step in enumerate(traj_astarte):
        raw_role = step.get("role", "user")
        if raw_role == "system":
            role = "user"
            content = f"[Lead Architect]: Next sprint on astarte-platform/astarte-device-sdk-python repository. Issue #98: Sending empty payloads in device.py raises exceptions during BSON serialization."
            cat = "fact"
            fact_id = "astarte_goal"
        elif raw_role == "ai":
            role = "assistant"
            content = f"[Coding Agent]: {str(step.get('text', '')).strip()}"
            cat = "fact" if any(w in content for w in ["device.py", "bson_payload", "payload", "Device", "pylint"]) else "filler"
            fact_id = f"astarte_ai_{i}"
        else:
            role = "user"
            text = str(step.get("text", "")).strip()
            content = f"[Developer / Terminal]: {text}"
            words = text.lower().split()
            if len(words) <= 5 and any(w in words for w in ["ok", "done", "bash-$"]):
                cat = "filler"
            else:
                cat = "fact"
            fact_id = f"astarte_dev_{i}"

        turns.append({
            "turn_id": turn_id,
            "role": role,
            "content": content,
            "category": cat,
            "fact_id": fact_id,
            "tags": ["session_astarte", "iot_sdk", "bson", "swe_real"],
        })
        turn_id += 1

    # -------------------------------------------------------------------------
    # Session 3: Architectural Refactoring, Contradictions, and Deprecations
    # -------------------------------------------------------------------------
    refactor_turns = [
        {
            "role": "user",
            "content": "[Engineering Manager]: Team architecture update: we are deprecating passing raw auth tuples directly to ServiceCaller.call(). Moving forward, all authenticated requests must use TokenAuthHandler with OAuth2 bearer token injection.",
            "category": "contradiction",
            "fact_id": "CONTRA_APIRON_AUTH",
            "tags": ["refactor", "apiron", "auth_update"],
        },
        {
            "role": "assistant",
            "content": "[Coding Agent]: Understood. I will ensure all future client invocations route through TokenAuthHandler with bearer token headers rather than raw auth tuples.",
            "category": "fact",
            "fact_id": "CONTRA_APIRON_AUTH_ACK",
            "tags": ["refactor", "apiron"],
        },
        {
            "role": "user",
            "content": "[DevOps Lead]: Infrastructure migration notice: the Astarte MQTT broker host has been migrated from legacy broker.astarte.internal:1883 to modern secure cluster mqtt-ssl.astarte.cloud:8883 with mutual TLS enforced.",
            "category": "contradiction",
            "fact_id": "CONTRA_ASTARTE_BROKER",
            "tags": ["refactor", "astarte", "mqtt_broker_update"],
        },
        {
            "role": "assistant",
            "content": "[Coding Agent]: Recorded. Updating configuration references to mqtt-ssl.astarte.cloud:8883 with mutual TLS enabled.",
            "category": "fact",
            "fact_id": "CONTRA_ASTARTE_BROKER_ACK",
            "tags": ["refactor", "astarte"],
        },
        {
            "role": "user",
            "content": "[Protocol Team]: Telemetry protocol spec update: empty payloads must now be encoded as compact JSON empty objects b'{}' instead of legacy raw BSON empty byte strings b''.",
            "category": "contradiction",
            "fact_id": "CONTRA_ASTARTE_PAYLOAD",
            "tags": ["refactor", "astarte", "payload_update"],
        },
        {
            "role": "assistant",
            "content": "[Coding Agent]: Confirmed. Serializer defaults updated to b'{}' for empty transmissions.",
            "category": "fact",
            "fact_id": "CONTRA_ASTARTE_PAYLOAD_ACK",
            "tags": ["refactor", "astarte"],
        },
        {
            "role": "user",
            "content": "[QA Lead]: Testing framework migration: we have deprecated nose across all repositories. All automated tests must now run under pytest with coverage: pytest --cov=apiron.",
            "category": "contradiction",
            "fact_id": "CONTRA_TEST_RUNNER",
            "tags": ["refactor", "testing", "ci_update"],
        },
        {
            "role": "assistant",
            "content": "[Coding Agent]: Got it. All continuous integration test pipelines will execute via pytest --cov=apiron.",
            "category": "fact",
            "fact_id": "CONTRA_TEST_RUNNER_ACK",
            "tags": ["refactor", "testing"],
        },
    ]

    for rf in refactor_turns:
        turns.append({
            "turn_id": turn_id,
            "role": rf["role"],
            "content": rf["content"],
            "category": rf["category"],
            "fact_id": rf["fact_id"],
            "tags": rf["tags"],
        })
        turn_id += 1

    # -------------------------------------------------------------------------
    # Comprehensive Evaluation Queries across Lexical, Semantic & Contradictions
    # -------------------------------------------------------------------------
    queries = [
        # --- Group 1: Lexical & Exact Code References (Tests BM25 & Hybrid Search) ---
        {
            "query_id": "swe_q01",
            "query": "Which class in apiron/client.py handles service call dispatching?",
            "target_facts": ["ServiceCaller", "class ServiceCaller"],
            "obsolete_facts": [],
            "category": "lexical_exact",
        },
        {
            "query_id": "swe_q02",
            "query": "What parameter was added to the ServiceCaller.call method signature in apiron?",
            "target_facts": ["auth=None", "auth"],
            "obsolete_facts": [],
            "category": "code_signature",
        },
        {
            "query_id": "swe_q03",
            "query": "In which file is the Device class defined in astarte-device-sdk-python?",
            "target_facts": ["astarte/device/device.py", "device.py"],
            "obsolete_facts": [],
            "category": "file_path",
        },
        {
            "query_id": "swe_q04",
            "query": "What line number in client.py was def call located around?",
            "target_facts": ["Line 112", "112"],
            "obsolete_facts": [],
            "category": "lexical_line_lookup",
        },
        {
            "query_id": "swe_q05",
            "query": "What helper method does ServiceCaller use to choose the target host?",
            "target_facts": ["choose_host", "cls.choose_host"],
            "obsolete_facts": [],
            "category": "code_method",
        },
        {
            "query_id": "swe_q06",
            "query": "What pylint disable comment is attached to the Device class header in astarte?",
            "target_facts": ["pylint: disable=too-many-instance-attributes", "too-many-instance-attributes"],
            "obsolete_facts": [],
            "category": "linting_directive",
        },
        {
            "query_id": "swe_q07",
            "query": "What initial assignment was made to bson_payload in device.py?",
            "target_facts": ['bson_payload = b""', 'b""'],
            "obsolete_facts": [],
            "category": "code_variable",
        },

        # --- Group 2: Semantic Reasoning & Problem Diagnosis ---
        {
            "query_id": "swe_q08",
            "query": "What was the core problem reported in apiron issue #8?",
            "target_facts": ["Add pass-through auth", "auth argument", "ServiceCaller.call"],
            "obsolete_facts": [],
            "category": "semantic_issue_reasoning",
        },
        {
            "query_id": "swe_q09",
            "query": "What issue occurred in astarte when transmitting empty payloads?",
            "target_facts": ["Sending empty payloads", "empty payloads", "BSON"],
            "obsolete_facts": [],
            "category": "semantic_issue_reasoning",
        },
        {
            "query_id": "swe_q10",
            "query": "What error did the agent encounter during its initial edit on device.py?",
            "target_facts": ["syntax error", "proper indentation", "indentation"],
            "obsolete_facts": [],
            "category": "error_diagnosis",
        },
        {
            "query_id": "swe_q11",
            "query": "Which test file was inspected or updated to verify apiron client authentication?",
            "target_facts": ["test_client.py", "tests"],
            "obsolete_facts": [],
            "category": "testing_workflow",
        },

        # --- Group 3: Contradictions & Temporal Evolution (Tests Memory Consolidation & Decay) ---
        {
            "query_id": "swe_q12",
            "query": "What is the new required authentication handler for ServiceCaller requests?",
            "target_facts": ["TokenAuthHandler", "bearer token"],
            "obsolete_facts": ["raw auth tuples", "basic auth"],
            "category": "contradiction_probe",
        },
        {
            "query_id": "swe_q13",
            "query": "What is the active Astarte MQTT broker host and port after infrastructure migration?",
            "target_facts": ["mqtt-ssl.astarte.cloud:8883", "mqtt-ssl.astarte.cloud"],
            "obsolete_facts": ["broker.astarte.internal:1883", "1883"],
            "category": "contradiction_probe",
        },
        {
            "query_id": "swe_q14",
            "query": "How are empty telemetry payloads encoded in the updated protocol specification?",
            "target_facts": ["b'{}'", "compact JSON empty objects", "compact JSON"],
            "obsolete_facts": ["b''", "raw BSON empty byte strings", "raw BSON"],
            "category": "contradiction_probe",
        },
        {
            "query_id": "swe_q15",
            "query": "What command should be used to run the automated test suite under the updated CI config?",
            "target_facts": ["pytest --cov=apiron", "pytest"],
            "obsolete_facts": ["nose"],
            "category": "contradiction_probe",
        },

        # --- Group 4: Hallucination & Out-of-Scope Probes (Must retrieve 0 irrelevant memories) ---
        {
            "query_id": "swe_q16",
            "query": "Which React frontend state management library was chosen?",
            "target_facts": [],
            "obsolete_facts": [],
            "category": "out_of_scope_probe",
        },
        {
            "query_id": "swe_q17",
            "query": "What is the deployment procedure for the Kubernetes Helm chart?",
            "target_facts": [],
            "obsolete_facts": [],
            "category": "out_of_scope_probe",
        },
    ]

    benchmark_dict = {
        "name": "SWE_Real_Trajectories_MultiSession",
        "description": f"Authentic software engineering benchmark derived from real SWE-agent developer trajectories (ithaka/apiron-8 and astarte-platform/astarte-device-sdk-python-98) totaling {len(turns)} turns across multi-file code editing, stack trace analysis, and architectural refactoring.",
        "user_id": "dev_engineer_swe",
        "turns": turns,
        "queries": queries,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_dict, f, indent=2)

    print(f"Successfully generated SWE real benchmark at {output_path}")
    print(f"Total turns: {len(turns)}, Total queries: {len(queries)}")
    return benchmark_dict


if __name__ == "__main__":
    prepare_swe_real_benchmark()
