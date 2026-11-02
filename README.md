# Training-free multi-agent detection of English and Arabic phishing emails

Code and evaluation scripts for the paper *Training-Free Multi-Agent LLM Detection of English and Arabic Phishing Emails: Ablation, Pre-Specified Configuration Selection, and Cross-Corpus Comparison with Trained Baselines*.

## The final system

| Agent | Implementation | Weight |
|---|---|---|
| Reasoning Agent | GPT-4o-mini (`gpt-4o-mini-2024-07-18`), temperature 0.3, chain-of-thought prompt with 11 worked examples in English and Arabic | 90% |
| Perspective Agent | Rule-based experts: P1 header and sender, P2 social engineering, P3 infrastructure vocabulary, P4 borderline phrases (English only). Score = mean of the experts | 10% |
| Pattern Agent | Keyword scoring (BM25 / TF-IDF) | 0% |

**Input sanitiser:** before the agents read an email, HTML comments and lines that address an AI or give classification instructions are removed; other emails are passed on unchanged.

**Decision:** `score = 0.90 × Reasoning + 0.10 × Perspective`; an email is phishing if `score ≥ 30`. The same weights and threshold are used for English and Arabic. There is no pre-filter, no override rule and no fallback score: if GPT-4o-mini fails after three attempts the system raises an error.

The configuration was selected on the development halves of each dataset (stratified 50/50 split, seed 2026) by a rule fixed in advance, frozen in `final_configuration/frozen_config.json`, and then evaluated once on the held-out test halves and on 3,000 previously unseen MeAJOR emails.

## Files

| File | Purpose | API cost |
|---|---|---|
| `AI_Phishing_Detection_System_Final.py` | The final system with a local web interface | GPT-4o-mini, about US$0.42 per 1,000 emails |
| `Select_And_Freeze_Configuration.py` | Development/test split, selection rule, frozen configuration | none (uses saved scores) |
| `Baseline_Classical.py` | TF-IDF + logistic regression / linear SVM: in-domain, cross-corpus, low-data | none |
| `Baseline_Transformer.py`, `Transformer_Baselines_Colab.ipynb` | Fine-tuned XLM-R and AraBERT (GPU, e.g. Google Colab) | none |
| `Pattern_BM25_TFIDF_Bilingual.py` | BM25 and TF-IDF keyword scoring for both languages | none |
| `Pareto_Unified.py` | Agent-combination Pareto analysis (C1–C7) on the test halves | none |
| `Repeat_Runs.py` | Two extra scoring runs of the 300-email datasets (mean ± SD) | about US$1 |
| `Robustness_Test.py` | Prompt injection, hidden instructions, benign text, paraphrase; input sanitiser | about US$0.75 |
| `Attack_Set.py` | Rebuilds the exact attacked emails used by the robustness test | none |
| `Baselines_Under_Attack.py` | TF-IDF baselines on the same attacked emails | none |
| `Transformer_Under_Attack.py`, `Colab_Attack_Test.ipynb` | XLM-R / AraBERT on the same attacked emails (GPU, e.g. Google Colab) | none |
| `Check_Perspective_And_Sanitiser.py` | Perspective-score reproducibility check; emails changed by the sanitiser | none |
| `Fix_Perspective_Cache.py` | Recomputes the saved Perspective scores with the final code (44 Arabic scores had been saved by an earlier version; no decision of the final configuration changed) and re-scores the 3 emails the sanitiser changes | under US$0.01 |
| `score_cache_reasoning_only/` | Per-email Reasoning and Perspective scores used in the paper | |
| `final_configuration/` | The frozen configuration | |
| `robustness_results/` | Per-email scores under attack and the cached paraphrases used by the robustness scripts | |
| `docs/DEVELOPMENT_HISTORY.md` | History of earlier development versions (not the final system) | |

## Running

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"          # never commit a key; .env is ignored by git
python3 AI_Phishing_Detection_System_Final.py
```

The evaluation scripts expect the dataset CSV files in the repository folder:

| Dataset | File | Source |
|---|---|---|
| Enron (curated, 29,767) | `Enron.csv` | Champa et al. (2024) |
| Nazario (1,565) | `Nazario.csv` | Nazario (2005) |
| Salloum English / Arabic (300 each) | `Salloum_English_300.csv`, `Salloum_Arabic_300.csv` | Salloum et al. (2023) |
| MeAJOR (3,000-email sample) | `meajor_cleaned_preprocessed.csv` | Mendes et al. (2025) |
| Synthetic Arabic critical-infrastructure emails (600) | `UAE_Synthetic_Arabic_Combined_600.csv` | this study |

Public datasets are not redistributed here; download them from their original sources.

## Citation

Please cite the paper (details to be added on publication).
