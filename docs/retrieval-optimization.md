# Retrieval Optimization

## Overview

Retrieval optimization was performed after establishing and benchmarking the baseline RAG pipeline.

The objective was not to maximize a single retrieval metric at any cost. The objective was to determine whether alternative retrieval strategies produced a meaningful improvement while preserving answer quality, groundedness, citation correctness, simplicity, and practical latency.

The final decision was to retain the baseline dense-vector retrieval strategy.

---

## Baseline Retrieval Configuration

The production retriever uses:

- **Vector database:** Qdrant
- **Similarity:** Cosine distance
- **Chunk size:** 1000
- **Chunk overlap:** 150
- **Production retrieval depth:** Top-5
- **Retrieval method:** Dense vector similarity search
- **Embedding:** Existing project embedding pipeline

The retriever also supports an optional collection name so experimental collections can be evaluated without changing the production collection.

---

## Candidate Recall Diagnostic

A Top-20 candidate recall diagnostic was used to determine whether relevant pages were entering the retrieval candidate pool.

| Candidate size | Average Recall |
|---|---:|
| Recall@5 | **91.67%** |
| Recall@10 | **94.17%** |
| Recall@20 | **97.50%** |

### Observations

Most benchmark questions achieved 100% recall at Top-5.

Two questions showed meaningful improvements with deeper candidate retrieval:

- The knowledge-base-enabled deployment question improved from 75% Recall@5 to 100% Recall@10.
- The RAG deployment prompt question improved from 66.67% Recall@5 to 100% Recall@20.

One broad monitoring question had a relevant expected page that was not retrieved even within Top-20. This indicates that simply increasing candidate depth cannot solve every semantic retrieval miss.

The diagnostic therefore showed that the baseline retriever is already strong, while some broad or multi-part questions remain difficult.

---

## Chunking Experiment

An alternative chunking configuration of **800 tokens with 100-token overlap** was evaluated against the baseline **1000/150** configuration.

The alternative configuration performed worse in the retrieval benchmark and was therefore not promoted to production.

The baseline 1000/150 configuration was retained.

---

## Reranking Experiment

A reranking experiment was evaluated using the retrieved candidates.

The reranked results produced:

- Hit Rate@5: **100.00%**
- Recall@5: **87.50%**
- Precision@5: **44.00%**
- MRR@5: **95.00%**

The reranker improved MRR, indicating better ordering of some relevant results, but reduced Recall@5 and Precision@5 compared with the baseline.

Because the overall trade-off was not a clear improvement, reranking remained an experimental strategy rather than becoming the production default.

---

## Multi-Query Experiment

Multi-query retrieval was evaluated to determine whether generating multiple query formulations improved retrieval coverage.

The experiment did not produce a meaningful improvement over the baseline, so multi-query retrieval was not promoted.

---

## Query Rewriting Experiment

Query rewriting was evaluated as part of the agent's recovery path rather than as the default retrieval strategy.

The production graph already uses query rewriting when the initial retrieval is judged insufficiently relevant.

This provides a targeted recovery mechanism without making every request more expensive.

---

## Final Retrieval Decision

The final production strategy remains:

```text
Question
   |
   v
Query Embedding
   |
   v
Qdrant Dense Search
   |
   v
Top-5 Results
   |
   v
Relevance Check
   |
   +--> Relevant ------> Answer Generation
   |
   +--> Not Relevant
             |
             v
        Query Rewrite
             |
             v
        Second Retrieval
             |
             +--> Relevant ------> Answer Generation
             |
             +--> Not Relevant --> Abstention
```

### Why the baseline was retained

The baseline provides a strong balance between:

- retrieval quality
- context size
- simplicity
- predictable behavior
- generation cost and latency
- compatibility with the existing LangGraph workflow

Increasing the production retrieval depth to Top-10 or Top-20 would improve candidate recall for some questions, but it would also introduce additional context that is not consistently useful.

Similarly, reranking produced a better MRR score but did not improve the overall retrieval profile enough to justify making it part of the production path.

The project therefore prioritizes **measured, explainable engineering trade-offs over optimization for a single metric**.

---

## Final Benchmark Context

The final 10-question benchmark was run after the retrieval and generation pipeline had been stabilized.

### Retrieval

| Metric | Result |
|---|---:|
| Hit Rate@5 | **100.00%** |
| Recall@5 | **91.67%** |
| Precision@5 | **48.00%** |
| MRR@5 | **88.33%** |

### Generation and citations

| Metric | Result |
|---|---:|
| Answer Rate | **100.00%** |
| Citation Rate | **100.00%** |
| Relevant Citation Rate | **100.00%** |
| Abstention Rate | **0.00%** |
| Valid Citation Rate | **100.00%** |
| Unsupported Citation | **0.00%** |

### Answer quality

| Metric | Result |
|---|---:|
| Correctness | **90.00%** |
| Groundedness | **100.00%** |
| Citation Correctness | **100.00%** |

These results support retaining the baseline retrieval strategy.

---

## Known Retrieval Limitations

The experiments exposed several realistic limitations:

1. Dense similarity does not guarantee that every relevant reference page appears in the Top-5.
2. Broad questions covering multiple concepts can require deeper candidate retrieval.
3. A relevant page can remain outside the Top-20 candidate set, meaning reranking alone cannot recover it.
4. Higher candidate depth is not automatically better because additional results can introduce less relevant context.
5. Retrieval optimization is coupled to generation quality; improving one retrieval metric does not necessarily improve the final answer.

These limitations are intentionally documented rather than hidden.

---

## Conclusion

Phase 9 established that the existing dense retrieval pipeline is sufficiently strong for the project's goals.

The final choice is therefore **not to over-engineer retrieval**.

The production system keeps the simpler 1000/150 dense retrieval configuration with Top-5 retrieval and uses LangGraph query rewriting as a targeted recovery mechanism.

Alternative approaches remain valuable as documented experiments and provide evidence for why they were not promoted.

**Phase 9 status: COMPLETE.**
