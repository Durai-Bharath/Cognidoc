# Stage 2 traceability: agent → class → file → tests

Each agent A1–A15 in *Construction.pdf* (Table 1, Algorithms 2–16) maps to exactly one
class. Algorithm 1 maps to the framework, and the post-construction `CommunityBuilder`
(DESIGN_DECISIONS Stage 2 #3) has its own row. `tests/construction/test_traceability.py`
parses the table below. It fails if any row names a class, file or test that does not
exist, or if an agent's `agent_id`/`algorithm` disagree with the row.

Test ids use the form `file::test_name`, relative to `tests/construction/`.

| Agent | Alg | Agent (PDF) | Class | File | Tests |
|---|---|---|---|---|---|
| — | 1 | Overall Agentic Graph Construction Framework | `AgenticGraphConstructionFramework` | `src/construction/graph.py` | `test_framework.py::test_stage1_output_builds_a_validated_graph`, `test_framework.py::test_reingesting_the_same_document_is_idempotent`, `test_framework.py::test_low_confidence_chunk_is_skipped`, `test_framework.py::test_rejects_follow_the_reject_branch`, `test_framework.py::test_one_failing_chunk_is_logged_and_skipped`, `test_framework.py::test_unexpected_crash_inside_a_chunk_is_isolated`, `test_framework.py::test_invalid_document_terminates_with_report`, `test_framework.py::test_graph_maps_algorithm_1_nodes`, `test_framework.py::test_cli_offline` |
| A1 | 2 | Document Ingestion Agent | `DocumentIngestionAgent` | `src/construction/agents/ingestion.py` | `test_ingestion_chunking.py::test_valid_document_is_registered_with_session`, `test_ingestion_chunking.py::test_document_id_is_content_hash_and_reingestion_is_detected`, `test_ingestion_chunking.py::test_invalid_schema_terminates_with_error_report`, `test_ingestion_chunking.py::test_metadata_mismatch_and_empty_document_are_rejected` |
| A2 | 3 | Semantic Chunking Agent | `SemanticChunkingAgent` | `src/construction/agents/chunking.py` | `test_ingestion_chunking.py::test_chunking_reuses_stage1_chunks_with_overlap_and_hierarchy`, `test_ingestion_chunking.py::test_chunking_splits_long_and_unchunked_sections_with_overlap`, `test_ingestion_chunking.py::test_chunking_tables_can_be_disabled_and_tiny_chunks_dropped`, `test_ingestion_chunking.py::test_chunk_helpers` |
| A3 | 4 | Triple Extraction Agent | `TripleExtractionAgent` | `src/construction/agents/triple_extraction.py` | `test_extraction_coref_confidence.py::test_triples_are_extracted_with_provenance_and_types`, `test_extraction_coref_confidence.py::test_triples_are_normalised_deduplicated_and_limited`, `test_extraction_coref_confidence.py::test_malformed_llm_output_is_retried`, `test_extraction_coref_confidence.py::test_llm_failure_is_a_typed_recoverable_error` |
| A4 | 5 | Coreference Resolution Agent | `CoreferenceResolutionAgent` | `src/construction/agents/coreference.py` | `test_extraction_coref_confidence.py::test_pronouns_and_document_references_are_resolved`, `test_extraction_coref_confidence.py::test_pronoun_uses_previous_chunk_subject_or_is_dropped`, `test_extraction_coref_confidence.py::test_abbreviations_resolve_to_canonical_names`, `test_extraction_coref_confidence.py::test_duplicate_mentions_are_merged`, `test_extraction_coref_confidence.py::test_resolution_creating_self_loop_is_dropped` |
| A5 | 6 | Confidence Evaluation Agent | `ConfidenceEvaluationAgent` | `src/construction/agents/confidence_evaluation.py` | `test_extraction_coref_confidence.py::test_confidence_threshold_accepts_and_discards`, `test_extraction_coref_confidence.py::test_ungrounded_triples_are_penalised`, `test_extraction_coref_confidence.py::test_no_surviving_triple_fails_the_chunk`, `test_extraction_coref_confidence.py::test_grounding_uses_aliases_and_context`, `test_framework.py::test_low_confidence_chunk_is_skipped` |
| A6 | 7 | Entity Resolution Agent | `EntityResolutionAgent` | `src/construction/agents/entity_resolution.py` | `test_resolution_normalization.py::test_exact_canonical_match_links_existing_node`, `test_resolution_normalization.py::test_similarity_at_or_above_delta_links`, `test_resolution_normalization.py::test_similarity_below_delta_creates_new_entity`, `test_resolution_normalization.py::test_type_incompatibility_blocks_linking`, `test_resolution_normalization.py::test_near_duplicate_new_mentions_in_one_batch_share_a_node` |
| A7 | 8 | Relation Normalization Agent | `RelationNormalizationAgent` | `src/construction/agents/relation_normalization.py` | `test_resolution_normalization.py::test_relations_map_by_exact_synonym_inverse_and_stem`, `test_resolution_normalization.py::test_llm_fallback_maps_unmatched_relation`, `test_resolution_normalization.py::test_unmatched_relation_is_rejected` |
| A8 | 9 | Conflict Detection Agent | `ConflictDetectionAgent` | `src/construction/agents/conflict_detection.py` | `test_conflict_decision_crud.py::test_new_when_relationship_absent`, `test_conflict_decision_crud.py::test_update_when_same_source_reasserts`, `test_conflict_decision_crud.py::test_merge_when_new_source_corroborates`, `test_conflict_decision_crud.py::test_conflict_on_functional_predicate_with_different_value`, `test_conflict_decision_crud.py::test_conflict_on_exclusive_predicates_in_kg_and_batch`, `test_conflict_decision_crud.py::test_reject_cases` |
| A9 | 10 | Decision Agent | `DecisionAgent` | `src/construction/agents/decision.py` | `test_conflict_decision_crud.py::test_decisions_map_statuses_and_log_conflict_records`, `test_framework.py::test_contradictory_triples_across_documents_are_logged_and_graph_unchanged` |
| A10 | 11 | Graph CRUD Agent | `GraphCRUDAgent` | `src/construction/agents/graph_crud.py` | `test_conflict_decision_crud.py::test_create_inserts_nodes_and_relationship`, `test_conflict_decision_crud.py::test_create_respects_inversion_and_existing_nodes`, `test_conflict_decision_crud.py::test_merge_adds_provenance_and_takes_max_confidence`, `test_conflict_decision_crud.py::test_update_changes_only_confidence_and_provenance`, `test_conflict_decision_crud.py::test_reject_discards_and_failed_operations_are_isolated`, `test_conflict_decision_crud.py::test_delete_retracts_source_then_removes_edge_and_orphans`, `test_framework.py::test_corroborating_document_merges` |
| A11 | 12 | Provenance Agent | `ProvenanceAgent` | `src/construction/agents/provenance.py` | `test_provenance_embedding_index.py::test_chunk_node_structural_links_and_mentions`, `test_provenance_embedding_index.py::test_missing_provenance_fields_are_repaired_from_the_chunk`, `test_provenance_embedding_index.py::test_incomplete_provenance_fails_and_is_flagged` |
| A12 | 13 | Embedding Agent | `EmbeddingAgent` | `src/construction/agents/embedding.py` | `test_provenance_embedding_index.py::test_entities_and_chunks_are_embedded_and_attached` |
| A13 | 14 | Vector Index Agent | `VectorIndexAgent` | `src/construction/agents/vector_index.py` | `test_provenance_embedding_index.py::test_vectors_are_upserted_mapped_and_verified`, `test_provenance_embedding_index.py::test_failed_verification_is_a_typed_error` |
| A14 | 15 | Construction Validation Agent | `ConstructionValidationAgent` | `src/construction/agents/construction_validation.py` | `test_validation_audit_communities.py::test_clean_graph_passes_validation`, `test_validation_audit_communities.py::test_ontology_violations_are_detected`, `test_validation_audit_communities.py::test_consistency_provenance_and_connectivity_failures`, `test_validation_audit_communities.py::test_vector_mapping_integrity`, `test_validation_audit_communities.py::test_invalid_entity_names_are_warnings` |
| A15 | 16 | Graph Health Audit Agent | `GraphHealthAuditAgent` | `src/construction/agents/health_audit.py` | `test_validation_audit_communities.py::test_audit_metrics_on_clean_graph`, `test_validation_audit_communities.py::test_audit_detects_orphans_duplicates_stale_and_low_confidence` |
| CB | — | Community summaries (post-step, not in the PDF) | `CommunityBuilder` | `src/construction/community.py` | `test_validation_audit_communities.py::test_communities_are_detected_summarised_linked_and_indexed`, `test_validation_audit_communities.py::test_rerun_skips_unchanged_and_rebuilds_changed_communities`, `test_validation_audit_communities.py::test_community_summary_falls_back_and_respects_settings` |

## Validation checks per agent

| Agent | Checks (error unless marked *warning*) |
|---|---|
| A1 | `document_schema`, `metadata_schema`, `model_parse`, `metadata_matches_document`, `has_content` |
| A2 | `unique_chunk_ids`, `non_empty`, `hierarchy_preserved` |
| A11 | `provenance_complete` |
| A13 | read-back verification: presence, `node_id` mapping, vector equality (typed `VectorIndexError`) |
| A14 | `ontology_compliance`, `node_edge_consistency`, `provenance_coverage`, `graph_connectivity`, `single_entity_component` (*warning*), `valid_entities_and_relations` (*warning*), `vector_mapping_integrity` |
| A15 | metrics, not pass/fail: orphans, duplicates, stale edges, components, low-confidence edges, `health_score` |

A3–A10 are decision branches rather than "Validate" blocks: schema-validated output with
retry (A3), pronoun, abbreviation and duplicate rules (A4), τ (A5), δ (A6), ontology
match or LLM fallback (A7), New/Merge/Update/Conflict/Reject (A8) and the operation
mapping (A9, A10). The tests above cover each branch.

## Infrastructure

| Component | File | Tests |
|---|---|---|
| `GraphStore`, `InMemoryGraphStore` (networkx) | `src/construction/stores/graph_store.py` | `test_infrastructure.py`, `test_store_contract.py::test_graph_store_contract` |
| `Neo4jGraphStore` | `src/construction/stores/neo4j_store.py` | `test_store_contract.py::test_graph_store_contract` (live), `test_store_contract.py::test_pipeline_runs_on_neo4j_and_qdrant` (live) |
| `VectorStore`, `InMemoryVectorStore` | `src/construction/stores/vector_store.py` | `test_infrastructure.py`, `test_store_contract.py::test_vector_store_contract` |
| `QdrantVectorStore` | `src/construction/stores/qdrant_store.py` | `test_store_contract.py::test_vector_store_contract` (live), `test_store_contract.py::test_pipeline_runs_on_neo4j_and_qdrant` (live) |
| `Ontology` (validated at startup) | `src/construction/ontology.py`, `ontology.yaml` | `test_infrastructure.py::test_default_ontology_matches_relations`, `test_infrastructure.py::test_invalid_ontology_is_rejected`, `test_framework.py::test_invalid_ontology_fails_at_startup` |
| `ConstructionConfig` (τ, δ, limits, backends) | `src/construction/config.py`, `construction.yaml` | `test_infrastructure.py::test_construction_config_loading` |
| Neo4j + Qdrant services | `docker-compose.yml` | live tests: `GRAPHRAG_LIVE_STORES=1 make test` |
| Reused from Stage 1 | `BaseAgent`, `LLMClient`/`MockLLM`, `Embedder`/`FakeEmbedder`, `ReviewQueue`, `schemas/*.json`, abbreviation repository (`extraction.agents.abbreviation`) | every test above |
