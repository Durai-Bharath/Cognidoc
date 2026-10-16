# Stage 3 traceability: agent → class → file → tests

Each agent A1–A15 in *Retrieval.pdf* (Table 1, Algorithms 2–16) maps to exactly one class.
Algorithm 1 maps to the framework. `tests/retrieval/test_traceability.py` parses the table
below. It fails if any row names a class, file or test that does not exist, or if an
agent's `agent_id`/`algorithm` disagree with the row.

The PDF prints no header for Algorithm 10. A9 (Answer Generation) is numbered 10 because
it sits between Algorithm 9 (A8) and Algorithm 11 (A10).

Test ids use the form `file::test_name`, relative to `tests/retrieval/`.

| Agent | Alg | Agent (PDF) | Class | File | Tests |
|---|---|---|---|---|---|
| — | 1 | Overall Agentic Graph Retrieval Framework | `AgenticGraphRetrievalFramework` | `src/retrieval/graph.py` | `test_framework.py::test_local_query_cites_real_chunks`, `test_framework.py::test_global_query_uses_community_summaries`, `test_framework.py::test_multi_hop_dependent_subqueries`, `test_framework.py::test_independent_subqueries_run_in_one_wave`, `test_framework.py::test_forced_failure_exhausts_retries_and_flags`, `test_framework.py::test_retry_recovers_on_a_later_attempt`, `test_framework.py::test_always_critique_can_invalidate_a_valid_answer`, `test_framework.py::test_agent_failures_are_isolated_and_the_run_terminates`, `test_framework.py::test_report_has_traces_and_persists`, `test_framework.py::test_mermaid_lists_algorithm_1_nodes`, `test_framework.py::test_api_query_and_report`, `test_framework.py::test_cli_query_and_report`, `test_framework.py::test_run_all_chains_three_stages` |
| A1 | 2 | Query Classification Agent | `QueryClassificationAgent` | `src/retrieval/agents/query_classification.py` | `test_query_agents.py::test_decide_evaluates_hybrid_first`, `test_query_agents.py::test_local_when_query_names_an_entity`, `test_query_agents.py::test_global_on_thematic_cues_without_entities`, `test_query_agents.py::test_hybrid_is_reachable`, `test_query_agents.py::test_simple_lookup_otherwise`, `test_query_agents.py::test_acronyms_are_expanded_before_entity_matching`, `test_query_agents.py::test_llm_judgement_overrides_heuristic`, `test_query_agents.py::test_llm_failure_or_empty_answer_keeps_heuristic` |
| A2 | 3 | Query Rewriting Agent | `QueryRewritingAgent` | `src/retrieval/agents/query_rewriting.py` | `test_query_agents.py::test_conversational_references_resolve_to_last_entity`, `test_query_agents.py::test_pronouns_stay_without_history`, `test_query_agents.py::test_acronyms_are_expanded`, `test_query_agents.py::test_spelling_is_corrected_to_corpus_vocabulary`, `test_query_agents.py::test_known_word_forms_are_left_alone`, `test_query_agents.py::test_enrichment_lists_entity_types`, `test_query_agents.py::test_llm_rewrite_accepted_only_if_entities_kept`, `test_framework.py::test_conversation_history_is_resolved_and_keyed` |
| A3 | 4 | Query Decomposition Agent | `QueryDecompositionAgent` | `src/retrieval/agents/query_decomposition.py` | `test_query_agents.py::test_simple_query_is_one_subquery`, `test_query_agents.py::test_dependency_chain_uses_placeholders`, `test_query_agents.py::test_independent_subqueries_share_a_wave`, `test_query_agents.py::test_validate_subqueries_detects_every_problem`, `test_query_agents.py::test_llm_decomposition_accepted_or_rejected` |
| A4 | 5 | Vector Retrieval Agent | `VectorRetrievalAgent` | `src/retrieval/agents/vector_retrieval.py` | `test_retrieval_agents.py::test_vector_retrieval_ranks_hydrated_chunks_and_entities`, `test_retrieval_agents.py::test_vector_hits_without_a_graph_node_are_dropped`, `test_retrieval_agents.py::test_min_score_and_entity_k_limit_results` |
| A5 | 6 | Graph Traversal Agent | `GraphTraversalAgent` | `src/retrieval/agents/graph_traversal.py` | `test_retrieval_agents.py::test_traversal_seeds_and_collects_cited_facts`, `test_retrieval_agents.py::test_zero_hops_returns_seeds_only`, `test_retrieval_agents.py::test_multi_hop_paths_are_scored_with_decay`, `test_retrieval_agents.py::test_traversal_respects_max_nodes` |
| A6 | 7 | Community Retrieval Agent | `CommunityRetrievalAgent` | `src/retrieval/agents/community_retrieval.py` | `test_retrieval_agents.py::test_community_retrieval_is_skipped_at_weight_zero`, `test_retrieval_agents.py::test_global_communities_ranked_by_relevance_times_weight`, `test_retrieval_agents.py::test_membership_and_query_type_weighting` |
| A7 | 8 | Evidence Aggregation Agent | `EvidenceAggregationAgent` | `src/retrieval/agents/evidence_aggregation.py` | `test_retrieval_agents.py::test_aggregation_builds_ranked_unified_evidence`, `test_retrieval_agents.py::test_feedback_confidence_lowers_fact_ranking`, `test_retrieval_agents.py::test_merge_items_deduplicates_and_unions_sources`, `test_retrieval_agents.py::test_aggregation_caps_items` |
| A8 | 9 | Context Construction Agent | `ContextConstructionAgent` | `src/retrieval/agents/context_construction.py` | `test_retrieval_agents.py::test_context_blocks_carry_citation_ids_and_provenance`, `test_retrieval_agents.py::test_token_budget_truncates_and_drops` |
| A9 | 10 | Answer Generation Agent | `AnswerGenerationAgent` | `src/retrieval/agents/answer_generation.py` | `test_answer_agents.py::test_empty_context_is_insufficient_without_llm_call`, `test_answer_agents.py::test_answer_is_grounded_with_markers`, `test_answer_agents.py::test_llm_insufficient_or_empty_answer_is_flagged`, `test_answer_agents.py::test_intermediate_mode_returns_short_answer`, `test_answer_agents.py::test_llm_failure_is_a_typed_error` |
| A10 | 11 | Citation Mapping Agent | `CitationMappingAgent` | `src/retrieval/agents/citation_mapping.py` | `test_answer_agents.py::test_markers_map_statements_to_chunks_nodes_edges_and_pages`, `test_answer_agents.py::test_unknown_markers_and_uncited_statements`, `test_answer_agents.py::test_split_statements_handles_marker_placement` |
| A11 | 12 | Faithfulness Validation Agent | `FaithfulnessValidationAgent` | `src/retrieval/agents/faithfulness_validation.py` | `test_answer_agents.py::test_supported_answer_is_valid_with_integrity_proof`, `test_answer_agents.py::test_hallucinated_number_is_unsupported_and_invalid`, `test_answer_agents.py::test_unresolvable_citation_fails_integrity`, `test_answer_agents.py::test_insufficient_answer_is_invalid`, `test_answer_agents.py::test_score_at_tau_is_valid`, `test_answer_agents.py::test_llm_judge_can_only_make_validation_stricter` |
| A12 | 13 | Self-Critique Agent | `SelfCritiqueAgent` | `src/retrieval/agents/self_critique.py` | `test_answer_agents.py::test_unsupported_claims_become_major_issues_with_implicated_knowledge`, `test_answer_agents.py::test_contradiction_is_detected`, `test_answer_agents.py::test_clean_answer_is_verified`, `test_answer_agents.py::test_llm_issues_must_name_known_statements`, `test_framework.py::test_always_critique_can_invalidate_a_valid_answer` |
| A13 | 14 | Feedback Agent | `FeedbackAgent` | `src/retrieval/agents/feedback.py` | `test_control_agents.py::test_bounded_step_never_crosses_floor_or_ceiling`, `test_control_agents.py::test_failed_answer_penalises_implicated_knowledge`, `test_control_agents.py::test_repeated_or_low_confidence_failures_are_flagged`, `test_control_agents.py::test_verified_answer_reinforces_supporting_knowledge`, `test_control_agents.py::test_reinforcement_can_be_disabled_and_state_persisted` |
| A14 | 15 | Retry Controller Agent | `RetryControllerAgent` | `src/retrieval/agents/retry_controller.py` | `test_control_agents.py::test_valid_answer_is_accepted`, `test_control_agents.py::test_escalation_sequence_per_attempt`, `test_control_agents.py::test_widen_communities_and_capped_steps_are_skipped`, `test_control_agents.py::test_no_possible_escalation_ends_the_run`, `test_framework.py::test_forced_failure_exhausts_retries_and_flags`, `test_framework.py::test_retry_recovers_on_a_later_attempt` |
| A15 | 16 | Cache Manager Agent | `CacheManagerAgent` | `src/retrieval/agents/cache_manager.py` | `test_control_agents.py::test_only_verified_answers_are_stored`, `test_control_agents.py::test_cache_hit_similar_hit_and_misses`, `test_control_agents.py::test_invalidation_by_element_and_on_graph_change`, `test_control_agents.py::test_purge_stale_after_construction`, `test_control_agents.py::test_cache_persists_and_can_be_disabled`, `test_control_agents.py::test_cache_evicts_oldest`, `test_framework.py::test_cache_hit_skips_retrieval_and_proves_citations`, `test_framework.py::test_penalised_knowledge_invalidates_cached_answers` |

## Algorithm 1 line → graph node

| Line | Algorithm 1 | Node(s) in `src/retrieval/graph.py` |
|---|---|---|
| 1 | Receive user query Q | `AgenticGraphRetrievalFramework.query` |
| — | Semantic cache check (brief: "check at the start") | `cache_lookup` (A15 `lookup`; hit → `generate_report`) |
| 2 | Query Classification [A1] | `query_classification` |
| 3 | Query Rewriting [A2] | `query_rewriting` |
| 4 | Query Decomposition [A3] | `query_decomposition` |
| 5 | foreach subquery qᵢ | `next_subquery` (dependency waves), `subquery_answers` (fills `{qN}`) |
| 6–9 | A4, A5, A6, A7 per subquery | `vector_retrieval` → `graph_traversal` → `community_retrieval` → `evidence_aggregation` (parallel within a wave) |
| 10–13 | A8, A9, A10, A11 | `context_construction` → `answer_generation` → `citation_mapping` → `faithfulness_validation` |
| 14–15 | Valid → return | `faithfulness_validation` → `feedback` (reinforce) → `cache_manager` |
| 16–19 | Else: A12, A13, A14 | `self_critique` → `feedback` (penalise; A15 `invalidate`) → `retry_controller` |
| 20 | Repeat until retry limit | `retry_controller` → `next_subquery` (restart at A4) or → `cache_manager` |
| 21 | Cache Manager [A15] | `cache_manager` (A15 `store`; verified answers only) |
| 22–23 | Generate R; return (A, C, R) | `generate_report` → `RetrievalResult` |

## Validation checks

| Agent | Checks (error unless marked *warning*) |
|---|---|
| A3 | `validate_subqueries`: ids `q1..qn` in order, dependencies on earlier ids only, placeholders backed by a dependency, ≤ `max_subqueries` (an LLM proposal that fails is rejected and flagged) |
| A11 | `has_claims`, `claims_supported` (*warning*), `citation_integrity`, `sufficient_evidence`, `faithfulness_score` (score ≥ τ) |
| A12 | `unsupported`, `citation`, `contradiction` issues (major), plus LLM issues |
| A14 | hard limit `retry.max_retries` (≤ 10); attempt counter strictly increases |

## Infrastructure

| Component | File | Tests |
|---|---|---|
| `RetrievalConfig` (τ, k, hops, weights, steps, limits) | `src/retrieval/config.py`, `retrieval.yaml` | every test (fixtures) |
| Data model (evidence, citations, report parts, fingerprints) | `src/retrieval/models.py` | every test |
| `SemanticCache` | `src/retrieval/agents/cache_manager.py` | `test_control_agents.py` (A15 rows) |
| Offline MockLLM handlers | `src/retrieval/testing.py` | every test |
| FastAPI `POST /query`, `GET /report/{id}` | `src/retrieval/api.py` | `test_framework.py::test_api_query_and_report` |
| CLI `graphrag query/report/serve/run-all` | `src/retrieval/cli.py`, `src/retrieval/run_all.py` | `test_framework.py::test_cli_query_and_report`, `test_framework.py::test_run_all_chains_three_stages`, `make run-example` |
| Reused | `BaseAgent`, `LLMClient`/`MockLLM`, `Embedder`/`FakeEmbedder`, `ReviewQueue` (Stage 1); `GraphStore`/`VectorStore`, Community nodes, `save_state` (Stage 2); `find_pairs` abbreviation matcher (Stage 1) | every test |
