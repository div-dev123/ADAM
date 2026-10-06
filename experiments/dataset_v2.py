"""Extended benchmark dataset with 42 turns across 5 domains."""

from experiments.dataset import BenchmarkDataset, ConversationTurn, EvalQuery


def get_extended_benchmark_dataset() -> BenchmarkDataset:
    """Constructs a rich 42-turn, 18-query benchmark dataset for ADAM v2 evaluation.

    Covers 5 distinct domains to expose ablation weaknesses:
      - Tech stack contradictions (tests consolidation + forgetting)
      - Location / personal context updates (tests contradiction resolution)
      - Infrastructure facts (tests factual recall with lexical queries)
      - Preferences / soft context (tests importance scoring)
      - Out-of-scope probes (tests hallucination suppression)

    Adversarial categories beyond the original v1 dataset:
      - Acronym-heavy queries (expose BM25 lexical advantage)
      - Multi-entity contradictions (expose consolidation precision)
      - Noisy filler interspersed throughout (expose noise filtering)
    """
    turns = [
        # Domain A: Initial Personal + Tech Context
        ConversationTurn(turn_id=1, role="user", content="Hey!", category="filler"),
        ConversationTurn(
            turn_id=2, role="user",
            content="I am building a high-throughput distributed event-processing platform using Apache Kafka and Golang.",
            category="fact", fact_id="tech_primary", tags=["tech", "backend", "initial"],
        ),
        ConversationTurn(turn_id=3, role="user", content="Sounds good.", category="filler"),
        ConversationTurn(
            turn_id=4, role="user",
            content="I live in San Francisco, California and work out of the Mission District.",
            category="fact", fact_id="location_v1", tags=["location", "initial"],
        ),
        ConversationTurn(
            turn_id=5, role="user",
            content="I prefer using PostgreSQL for all persistent relational data and Redis for caching.",
            category="fact", fact_id="database_stack", tags=["database", "infrastructure"],
        ),
        ConversationTurn(
            turn_id=6, role="user",
            content="Our Kafka cluster bootstraps at kafka-cluster.prod.internal:9092,9093,9094.",
            category="fact", fact_id="kafka_config", tags=["infrastructure", "config", "acronym"],
        ),
        ConversationTurn(turn_id=7, role="user", content="Great, thanks.", category="filler"),
        # Domain B: Preferences and Soft Context
        ConversationTurn(
            turn_id=8, role="user",
            content="I always drink black coffee in the morning before any deep work sessions.",
            category="preference", fact_id="morning_routine", tags=["preference", "routine"],
        ),
        ConversationTurn(
            turn_id=9, role="user",
            content="My favourite IDE is VSCode, specifically with the Vim keybinding extension.",
            category="preference", fact_id="ide_preference", tags=["preference", "tools"],
        ),
        ConversationTurn(
            turn_id=10, role="user",
            content="I prefer async programming patterns over synchronous blocking I/O wherever possible.",
            category="preference", fact_id="coding_style", tags=["preference", "programming"],
        ),
        ConversationTurn(turn_id=11, role="user", content="OK cool.", category="filler"),
        # Domain C: Infrastructure and SLAs
        ConversationTurn(
            turn_id=12, role="user",
            content="Our production API gateway enforces a strict p99 latency SLA of 30ms for all public endpoints.",
            category="fact", fact_id="sla_latency", tags=["sla", "requirements", "api"],
        ),
        ConversationTurn(
            turn_id=13, role="user",
            content="We run our Kubernetes cluster on GKE in us-central1 region with autoscaling enabled.",
            category="fact", fact_id="k8s_config", tags=["infrastructure", "cloud", "k8s"],
        ),
        ConversationTurn(
            turn_id=14, role="user",
            content="All microservices follow our internal CI/CD pipeline defined in GitHub Actions.",
            category="fact", fact_id="cicd_config", tags=["devops", "infrastructure"],
        ),
        # Domain D: First Wave of Contradictions
        ConversationTurn(turn_id=15, role="user", content="Good morning!", category="filler"),
        ConversationTurn(
            turn_id=16, role="user",
            content="We switched from Golang to Rust for our core event-processing engine. The team completed the migration last week.",
            category="contradiction", fact_id="tech_primary_update", tags=["tech", "backend", "contradiction"],
        ),
        ConversationTurn(
            turn_id=17, role="user",
            content="Actually, I relocated to Austin, Texas. San Francisco got too expensive.",
            category="contradiction", fact_id="location_v2", tags=["location", "contradiction"],
        ),
        ConversationTurn(turn_id=18, role="user", content="Yep.", category="filler"),
        ConversationTurn(
            turn_id=19, role="user",
            content="We are also moving from PostgreSQL to CockroachDB for its distributed SQL capabilities.",
            category="contradiction", fact_id="database_stack_update", tags=["database", "contradiction"],
        ),
        # Domain E: Reinforcement of Updated Facts
        ConversationTurn(
            turn_id=20, role="user",
            content="Reminder: our event processor is Rust-based. We no longer use Go for this component.",
            category="repeat", fact_id="tech_primary_update_repeat", tags=["tech", "repeat", "update"],
        ),
        ConversationTurn(
            turn_id=21, role="user",
            content="Just confirming: I am now based in Austin, not San Francisco.",
            category="repeat", fact_id="location_v2_repeat", tags=["location", "repeat"],
        ),
        ConversationTurn(turn_id=22, role="user", content="Thanks!", category="filler"),
        # Domain F: New Facts After Contradictions
        ConversationTurn(
            turn_id=23, role="user",
            content="We are deploying a new ML inference service using Triton Inference Server on GPU nodes.",
            category="fact", fact_id="ml_infra", tags=["ml", "infrastructure", "new"],
        ),
        ConversationTurn(
            turn_id=24, role="user",
            content="Our observability stack is Prometheus for metrics, Loki for logs, and Grafana for dashboards.",
            category="fact", fact_id="observability", tags=["observability", "devops", "acronym"],
        ),
        ConversationTurn(
            turn_id=25, role="user",
            content="I now prefer Zed editor over VSCode since it is faster and written in Rust.",
            category="contradiction", fact_id="ide_update", tags=["preference", "contradiction"],
        ),
        ConversationTurn(turn_id=26, role="user", content="OK.", category="filler"),
        ConversationTurn(
            turn_id=27, role="user",
            content="We now enforce zero-downtime deployments using blue-green strategies via Argo Rollouts.",
            category="fact", fact_id="deployment_strategy", tags=["devops", "strategy"],
        ),
        # Domain G: Acronym-Heavy Lexical Facts
        ConversationTurn(
            turn_id=28, role="user",
            content="Our OIDC provider is Okta, configured with PKCE for all OAuth2.0 flows.",
            category="fact", fact_id="auth_config", tags=["security", "auth", "acronym"],
        ),
        ConversationTurn(
            turn_id=29, role="user",
            content="The service mesh is Istio with mTLS enforced between all pod-to-pod communications.",
            category="fact", fact_id="service_mesh", tags=["infrastructure", "security", "acronym"],
        ),
        ConversationTurn(turn_id=30, role="user", content="Sure.", category="filler"),
        ConversationTurn(
            turn_id=31, role="user",
            content="Our internal build tool is Bazel, with remote caching on BuildBuddy.",
            category="fact", fact_id="build_tool", tags=["devops", "build"],
        ),
        # Domain H: Second Wave of Contradictions
        ConversationTurn(
            turn_id=32, role="user",
            content="We dropped CockroachDB and reverted to PostgreSQL. CockroachDB licensing costs were too high.",
            category="contradiction", fact_id="database_revert", tags=["database", "contradiction"],
        ),
        ConversationTurn(
            turn_id=33, role="user",
            content="We are migrating from GKE to AWS EKS in us-east-1 for better cost efficiency.",
            category="contradiction", fact_id="k8s_config_update", tags=["infrastructure", "cloud", "contradiction"],
        ),
        ConversationTurn(turn_id=34, role="user", content="Got it.", category="filler"),
        # Domain I: Late Additions
        ConversationTurn(
            turn_id=35, role="user",
            content="My team consists of 8 engineers: 5 backend, 2 SRE, and 1 ML engineer.",
            category="fact", fact_id="team_composition", tags=["team", "org"],
        ),
        ConversationTurn(
            turn_id=36, role="user",
            content="Our weekly team sync is every Tuesday at 10am PT.",
            category="fact", fact_id="meeting_schedule", tags=["team", "schedule"],
        ),
        ConversationTurn(
            turn_id=37, role="user",
            content="Reminder: all Rust code must use tokio for async runtime. No std async.",
            category="fact", fact_id="rust_async_rule", tags=["tech", "standards"],
        ),
        ConversationTurn(turn_id=38, role="user", content="Awesome.", category="filler"),
        ConversationTurn(
            turn_id=39, role="user",
            content="I switched my morning ritual: I now prefer matcha over black coffee.",
            category="contradiction", fact_id="morning_routine_update", tags=["preference", "contradiction"],
        ),
        ConversationTurn(
            turn_id=40, role="user",
            content="Our new p99 latency target for all APIs has been tightened to 20ms.",
            category="contradiction", fact_id="sla_latency_update", tags=["sla", "contradiction"],
        ),
        ConversationTurn(turn_id=41, role="user", content="Bye for now.", category="filler"),
        ConversationTurn(
            turn_id=42, role="user",
            content="Oh and one last thing: we use Terraform for all infrastructure-as-code provisioning.",
            category="fact", fact_id="iac_tool", tags=["devops", "iac"],
        ),
    ]

    queries = [
        # Contradiction Resolution Queries
        EvalQuery(
            query_id="q01_backend_language",
            query="What programming language powers our core event-processing engine?",
            target_facts=["Rust", "switched to Rust", "Rust-based"],
            obsolete_facts=["Golang", "Go for our core", "no longer use Go"],
            category="contradiction_resolution",
        ),
        EvalQuery(
            query_id="q02_location",
            query="Where do I currently live and work?",
            target_facts=["Austin", "Austin, Texas"],
            obsolete_facts=["San Francisco", "Mission District", "California"],
            category="contradiction_resolution",
        ),
        EvalQuery(
            query_id="q03_database",
            query="What database system do we use for relational persistence?",
            target_facts=["PostgreSQL", "reverted to PostgreSQL"],
            obsolete_facts=["CockroachDB"],
            category="contradiction_resolution",
        ),
        EvalQuery(
            query_id="q04_ide",
            query="What is my preferred code editor or IDE?",
            target_facts=["Zed", "Zed editor"],
            obsolete_facts=["VSCode"],
            category="contradiction_resolution",
        ),
        EvalQuery(
            query_id="q05_morning_drink",
            query="What do I drink in the morning before work?",
            target_facts=["matcha"],
            obsolete_facts=["black coffee"],
            category="contradiction_resolution",
        ),
        EvalQuery(
            query_id="q06_latency_sla",
            query="What is our current p99 API latency SLA requirement?",
            target_facts=["20ms", "tightened to 20ms"],
            obsolete_facts=["30ms"],
            category="contradiction_resolution",
        ),
        EvalQuery(
            query_id="q07_cloud_platform",
            query="Which cloud platform runs our Kubernetes cluster?",
            target_facts=["AWS EKS", "us-east-1", "EKS"],
            obsolete_facts=["GKE", "us-central1", "Google"],
            category="contradiction_resolution",
        ),
        # Factual Recall Queries (Lexical + Semantic Mix)
        EvalQuery(
            query_id="q08_kafka_address",
            query="What are the Kafka bootstrap server addresses for our production cluster?",
            target_facts=["kafka-cluster.prod.internal:9092", "9093", "9094"],
            obsolete_facts=[],
            category="factual_recall",
        ),
        EvalQuery(
            query_id="q09_observability",
            query="What tools make up our observability and monitoring stack?",
            target_facts=["Prometheus", "Loki", "Grafana"],
            obsolete_facts=[],
            category="factual_recall",
        ),
        EvalQuery(
            query_id="q10_auth",
            query="What is our OIDC and OAuth2 authentication configuration?",
            target_facts=["Okta", "PKCE", "OAuth2"],
            obsolete_facts=[],
            category="factual_recall",
        ),
        EvalQuery(
            query_id="q11_service_mesh",
            query="What service mesh do we run and what security policy does it enforce?",
            target_facts=["Istio", "mTLS"],
            obsolete_facts=[],
            category="factual_recall",
        ),
        EvalQuery(
            query_id="q12_build_tool",
            query="What build system does the engineering team use?",
            target_facts=["Bazel", "BuildBuddy"],
            obsolete_facts=[],
            category="factual_recall",
        ),
        EvalQuery(
            query_id="q13_team_size",
            query="How many engineers are on my team and what are their roles?",
            target_facts=["8 engineers", "5 backend", "2 SRE", "1 ML"],
            obsolete_facts=[],
            category="factual_recall",
        ),
        EvalQuery(
            query_id="q14_iac",
            query="What tool do we use for infrastructure as code provisioning?",
            target_facts=["Terraform"],
            obsolete_facts=[],
            category="factual_recall",
        ),
        # Preference / Soft Context Queries
        EvalQuery(
            query_id="q15_coding_style",
            query="What programming style does the team prefer for I/O operations?",
            target_facts=["async", "asynchronous", "non-blocking"],
            obsolete_facts=[],
            category="preference_recall",
        ),
        EvalQuery(
            query_id="q16_rust_standard",
            query="What async runtime must all Rust code use?",
            target_facts=["tokio"],
            obsolete_facts=[],
            category="factual_recall",
        ),
        EvalQuery(
            query_id="q17_deployment",
            query="How do we handle production deployments without downtime?",
            target_facts=["blue-green", "Argo Rollouts", "zero-downtime"],
            obsolete_facts=[],
            category="factual_recall",
        ),
        # Hallucination Control
        EvalQuery(
            query_id="q18_out_of_scope",
            query="What is my favourite restaurant in San Francisco?",
            target_facts=[],
            obsolete_facts=[],
            category="hallucination_control",
        ),
    ]

    return BenchmarkDataset(
        name="ADAM Extended Research Benchmark v2",
        description=(
            "A rich 42-turn, 18-query multi-domain benchmark covering tech stack contradictions, "
            "location updates, infrastructure facts, preference tracking, acronym-heavy lexical queries, "
            "and hallucination control probes. Designed to expose differentiable signal between ablation "
            "configurations across multiple retrieval strategies."
        ),
        user_id="research_eval_user_v2",
        turns=turns,
        queries=queries,
    )
