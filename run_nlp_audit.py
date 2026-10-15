"""
run_nlp_audit.py — Comprehensive NLP Pipeline Audit & Evaluation Harness
Tests all core NLP subsystems:
1. Lexical Tokenization & Normalization
2. Corpus-Mined Lexical Query Expansion
3. Query Intent & Named Entity Extraction (State Acronyms & Category Density)
4. Okapi BM25 Lexical Retrieval
5. Dense Semantic Vector Retrieval (all-MiniLM-L6-v2, 384-d normalized embeddings)
6. Hybrid Reciprocal Rank Fusion (Weighted RRF, k=60)
7. Full Evaluation across IR metrics: Top-1 Accuracy, Recall@3, Recall@5, MRR@5, MAP@5, Latency
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
import re

# Fix Windows cp1252 output encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import search_engine

def run_audit():
    print("=" * 80)
    print("  MY YOJANA SAARTHI — COMPREHENSIVE NLP AUDIT & PERFORMANCE BENCHMARK")
    print("=" * 80)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. SUBSYSTEM AUDIT: Index & Artifacts Integrity
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 1] Index & Model Artifacts Integrity Check...")
    t0 = time.time()
    df = search_engine.load_index()
    load_time = time.time() - t0

    emb = search_engine._embeddings
    model = search_engine._model
    bm25 = search_engine._bm25
    syn_map = search_engine._synonym_map

    print(f"  ✓ Corpus size: {len(df):,} schemes")
    print(f"  ✓ Embeddings shape: {emb.shape} | dtype: {emb.dtype}")
    
    # Check L2 normalization of precomputed vectors
    norms = np.linalg.norm(emb, axis=1)
    norm_min, norm_max, norm_mean = norms.min(), norms.max(), norms.mean()
    is_normalized = np.allclose(norms, 1.0, atol=1e-3)
    print(f"  ✓ Vector L2 normalization: {is_normalized} (mean={norm_mean:.4f}, min={norm_min:.4f}, max={norm_max:.4f})")
    
    # Check model architecture
    print(f"  ✓ Transformer encoder: {model.tokenizer.name_or_path} (dim={emb.shape[1]})")
    
    # Check synonym map
    print(f"  ✓ Synonym Lexicon: {len(syn_map):,} entries loaded")
    
    # Check BM25 index
    print(f"  ✓ BM25 Okapi Index: {bm25.corpus_size:,} documents indexed")
    print(f"  ✓ Index Load Time: {load_time:.2f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. SUBSYSTEM AUDIT: Query Expansion & Lexical Semantics
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 2] Query Expansion (Corpus-Mined Synonym Enrichment) Test...")
    test_expansion_queries = [
        "farmer loan",
        "scholarship disabled students",
        "solar rooftop",
        "laptop assistance college",
        "widow pension",
        "small business rural startup"
    ]
    expansion_results = []
    for q in test_expansion_queries:
        exp = search_engine.expand_query(q)
        q_tokens = len(q.split())
        exp_tokens = len(exp.split())
        expansion_results.append({
            "original_query": q,
            "expanded_query": exp,
            "orig_tokens": q_tokens,
            "expanded_tokens": exp_tokens,
            "ratio": exp_tokens / q_tokens
        })
        print(f"  Query:    '{q}' ({q_tokens} tokens)")
        print(f"  Expanded: '{exp}' ({exp_tokens} tokens, {exp_tokens/q_tokens:.1f}x density)\n")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. SUBSYSTEM AUDIT: Zero-Click Intent & Entity Extraction (NER)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 3] Entity Extraction & Intent Classification Test...")
    intent_test_cases = [
        ("tractor subsidy in UP", "Uttar Pradesh", "Agriculture, Rural & Environment"),
        ("disabled scholarship Maharashtra", "Maharashtra", "Education & Learning"),
        ("hospital treatment for cancer in WB", "West Bengal", "Health & Wellness"),
        ("pucca house awas in Bihar", "Bihar", "Housing & Shelter"),
        ("widow pension in Delhi", "Delhi", "Social welfare & Empowerment"),
        ("solar rooftop subsidy in Gujarat", "Gujarat", "Utility & Sanitation"),
        ("MSME loan for street vendor in TN", "Tamil Nadu", "Business & Entrepreneurship"),
        ("iti apprentice job training in MP", "Madhya Pradesh", "Skills & Employment"),
    ]

    intent_correct = 0
    for q, exp_state, exp_cat in intent_test_cases:
        parsed = search_engine.parse_query_intent(q)
        state_match = (parsed["detected_state"] == exp_state)
        cat_match = (parsed["detected_category"] == exp_cat)
        if state_match and cat_match:
            intent_correct += 1
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        print(f"  {status} Query: '{q}' -> State: {parsed['detected_state']} (expected {exp_state}) | Cat: {parsed['detected_category']} (expected {exp_cat})")

    intent_acc = intent_correct / len(intent_test_cases)
    print(f"  Intent Accuracy: {intent_acc:.1%} ({intent_correct}/{len(intent_test_cases)})")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. SUBSYSTEM AUDIT: Comparative Information Retrieval Evaluation
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 4] Ground-Truth IR Evaluation (eval_queries.json)...")
    eval_path = os.path.join(os.path.dirname(__file__), "eval_queries.json")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    print(f"  Evaluating {len(eval_data)} standard welfare benchmark queries across 4 IR methods...\n")

    methods = {
        "1. Naive Keyword Overlap": lambda q, k: search_engine.keyword_search(q, k=k),
        "2. Okapi BM25 Lexical": lambda q, k: search_engine.bm25_search(q, k=k),
        "3. Semantic Vector (MiniLM)": lambda q, k: search_engine.semantic_search(q, k=k),
        "4. Hybrid Fusion (BM25 + Sem RRF)": lambda q, k: search_engine.hybrid_search(q, k=k),
    }

    benchmark_results = {}

    for name, fn in methods.items():
        top1_hits = 0
        rec3_hits = 0
        rec5_hits = 0
        rr_sum = 0.0
        ap_sum = 0.0
        latencies = []

        for item in eval_data:
            q = item["query"]
            relevant = set(item["relevant_schemes"])

            t_start = time.perf_counter()
            res = fn(q, k=5)
            latencies.append((time.perf_counter() - t_start) * 1000)

            retrieved = res["scheme_id"].tolist() if "scheme_id" in res.columns else []

            # Top-1
            if retrieved and retrieved[0] in relevant:
                top1_hits += 1

            # Recall@3
            if any(r in relevant for r in retrieved[:3]):
                rec3_hits += 1

            # Recall@5
            if any(r in relevant for r in retrieved[:5]):
                rec5_hits += 1

            # Reciprocal Rank (MRR@5)
            rr = 0.0
            for rank, rid in enumerate(retrieved[:5], start=1):
                if rid in relevant:
                    rr = 1.0 / rank
                    break
            rr_sum += rr

            # Average Precision (MAP@5)
            hits = 0
            prec_sum = 0.0
            for rank, rid in enumerate(retrieved[:5], start=1):
                if rid in relevant:
                    hits += 1
                    prec_sum += hits / rank
            ap = prec_sum / min(len(relevant), 5) if relevant else 0.0
            ap_sum += ap

        n = len(eval_data)
        top1_acc = top1_hits / n
        rec3 = rec3_hits / n
        rec5 = rec5_hits / n
        mrr5 = rr_sum / n
        map5 = ap_sum / n
        avg_lat = np.mean(latencies)
        p95_lat = np.percentile(latencies, 95)

        benchmark_results[name] = {
            "Top-1 Accuracy": f"{top1_acc:.1%}",
            "Recall@3": f"{rec3:.1%}",
            "Recall@5": f"{rec5:.1%}",
            "MRR@5": f"{mrr5:.3f}",
            "MAP@5": f"{map5:.3f}",
            "Avg Latency (ms)": f"{avg_lat:.2f} ms",
            "P95 Latency (ms)": f"{p95_lat:.2f} ms",
            "raw_rec5": rec5,
            "raw_mrr": mrr5,
            "raw_lat": avg_lat
        }

    bench_df = pd.DataFrame(benchmark_results).T
    cols_display = ["Top-1 Accuracy", "Recall@3", "Recall@5", "MRR@5", "MAP@5", "Avg Latency (ms)", "P95 Latency (ms)"]
    print(bench_df[cols_display].to_string())

    # ──────────────────────────────────────────────────────────────────────────
    # 5. REAL-WORLD DEMOGRAPHIC SCENARIO TESTING (Edge Cases & NLP Stress Test)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("[PHASE 5] Real-World Demographic Scenario Testing & Portal Links...")
    print("=" * 80)

    stress_queries = [
        "free laptop for college students in Goa",
        "subsidy for rooftop solar panels in Tamil Nadu",
        "scholarship for disabled students in higher education",
        "kisan credit card and crop loan in Uttar Pradesh",
        "financial help for widow with school going children in Delhi",
        "I want to start a small business in rural area under startup policy",
        "free medical treatment for cancer patient in West Bengal",
        "pension scheme for retired old age senior citizens above 60"
    ]

    scenario_audit = []
    for q in stress_queries:
        intent = search_engine.parse_query_intent(q)
        res = search_engine.hybrid_search(q, k=3)
        top_scheme = res.iloc[0] if len(res) > 0 else None
        
        if top_scheme is not None:
            slug = str(top_scheme.get("slug", top_scheme.get("scheme_id", "")))
            portal_url = f"https://www.myscheme.gov.in/schemes/{slug}" if slug and slug != "nan" else "N/A"
            item = {
                "query": q,
                "detected_state": intent.get("detected_state"),
                "detected_category": intent.get("detected_category"),
                "top_scheme_name": top_scheme["scheme_name"],
                "rrf_score": float(top_scheme["score"]),
                "semantic_score": float(top_scheme.get("semantic_score", 0.0)),
                "bm25_score": float(top_scheme.get("bm25_score", 0.0)),
                "state": top_scheme.get("state", "All"),
                "portal_url": portal_url,
                "has_benefits": pd.notna(top_scheme.get("benefits")) and str(top_scheme.get("benefits")) != "nan",
                "has_eligibility": pd.notna(top_scheme.get("eligibility")) and str(top_scheme.get("eligibility")) != "nan",
            }
            scenario_audit.append(item)
            print(f"\nQuery: \"{q}\"")
            print(f"  → Intent: State={item['detected_state']} | Cat={item['detected_category']}")
            print(f"  → Top Scheme: {item['top_scheme_name']}")
            print(f"  → RRF: {item['rrf_score']:.4f} (Sem: {item['semantic_score']:.3f}, BM25: {item['bm25_score']:.1f}) | State: {item['state']}")
            print(f"  → Portal Slug: {slug} | URL: {portal_url}")
            print(f"  → Metadata Richness: Benefits={item['has_benefits']} | Eligibility={item['has_eligibility']}")

    # Save complete audit report to JSON
    report_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "index_stats": {
            "corpus_size": len(df),
            "embedding_dim": int(emb.shape[1]),
            "embeddings_dtype": str(emb.dtype),
            "vector_normalized": bool(is_normalized),
            "synonym_map_entries": len(syn_map),
            "bm25_corpus_size": bm25.corpus_size,
        },
        "expansion_sample": expansion_results,
        "intent_accuracy": intent_acc,
        "ir_benchmarks": {k: {col: v[col] for col in cols_display} for k, v in benchmark_results.items()},
        "scenario_audit": scenario_audit
    }

    out_json = os.path.join(os.path.dirname(__file__), "results", "nlp_audit_report.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_output, f, indent=2)

    print(f"\n\nAudit complete! Full benchmark JSON saved to: {out_json}")
    return report_output

if __name__ == "__main__":
    run_audit()
