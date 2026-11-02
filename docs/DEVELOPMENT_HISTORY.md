# Development history of the detection system (for the thesis)

This is the original module header of the development versions (v36 to v50_TFIDF), kept for the record.
It describes earlier configurations and is NOT a description of the final system.

```text
AI Phishing Detection System - FINAL (built from v50_TFIDF, MOM 40 Action 1)

FINAL CONFIGURATION (frozen by the pre-specified selection protocol, MOM 39/40)
    Score = 0.00 x Pattern + 0.90 x Reasoning + 0.10 x Perspective
    Phishing if Score >= 30. Same weights and threshold for English and Arabic.
    No override rules and no rule-based pre-filter in the decision.

CHANGES FROM v50_TFIDF (audited; reported results are unchanged)
    1. Weights 0/90/10 and threshold 30 for both languages (were English 70/30 T=30,
       Arabic 75/25 T=25), and the OR-logic override (Reasoning >= 55 or
       Perspective >= 65 forced a phishing decision) removed, so the code makes
       exactly the decision that was evaluated.
    2. P4 (borderline expert) is now an explicit rule-based expert. The earlier
       GPT-4o-mini call in P4 never executed (it referenced an undefined variable
       and a non-existent client attribute), so every result used these rules.
       The dead call is removed; Perspective scores are identical to v50_TFIDF.
    3. No silent fallback scoring: if the Reasoning Agent (GPT-4o-mini) fails, it
       retries, then raises ReasoningAgentError. It never substitutes a keyword
       score labelled as a GPT score.
    4. Trusted-sender pre-filter switched off by default
       (SystemConfig.USE_TRUSTED_SENDER_PREFILTER = False): it was not part of the
       evaluated system, and substring matching on the sender can be spoofed.
    5. The OpenAI key is read only from the OPENAI_API_KEY environment variable
       (or a local .env file). No key is stored in this file.
    6. One regular expression in P1 used ' *+' (valid only from Python 3.11); it is now ' *',
       which matches exactly the same text, so the file also runs on Python 3.9.
    Reasoning Agent prompt: chain-of-thought with 11 worked examples (unchanged).

---------------------------------------------------------------------------
History of the v50_TFIDF base:
Multi-Agent Architecture for Critical Infrastructure Protection

NEW ENHANCEMENTS IN v50_TFIDF (June 2026):
1. ✅ PATTERN AGENT REMOVED - replaced by stronger P4 Borderline Expert
      Rationale: Ablation studies (Config 6) consistently show that the
      Reasoning + Perspective combination outperforms the full 3-agent system
      on precision (Enron FP: 14→1, Nazario FP: 15→2).
      The Pattern Agent (TF-IDF) was the primary source of false positives,
      misidentifying legitimate corporate vocabulary as phishing signals.
      For Arabic emails, TF-IDF contribution was already reduced to 10%
      (v47c) due to inability to process Arabic text - removal eliminates
      this architectural inconsistency entirely.
      Reference: Ablation Study Config 6, v47c (May 2026).

2. ✅ P4 BORDERLINE EXPERT - upgraded from keyword-based (v49) to full
      GPT-4o-mini reasoning with bilingual Arabic/English support.
      P4 now makes a dedicated GPT API call with a specialised prompt
      targeting subtle phishing signals not detectable by standard agents.
      Inspired by Heiding et al. (2024): LLM-generated phishing exploits
      emotional manipulation achieving 30-44% click-through rates, proving
      that LLMs understand subtle phishing cues (IEEE Access, 12, 42131-42146).
      P4 covers 7 signal categories in both English and Arabic.

3. ✅ RECALIBRATED WEIGHTS - without Pattern Agent:
      English: R=70%, Pe=30%
      Arabic:  R=70%, Pe=30%  (R kept at v47c level; Pe absorbs Pattern weight)
      Threshold: T=30 (English AND Arabic)
5. ✅ P4 DISABLED FOR ARABIC (v50 fix3):
      Three consecutive Arabic test runs showed FP=28-31 regardless of
      threshold (T=25, T=27, T=30), confirming P4 GPT structurally
      over-detects Arabic legitimate emails via S5/S6 signal bias.
      Arabic emails use P1+P2+P3 aggregation (÷3) as in v47c.
      English emails retain GPT P4 (÷4 with weighted signals).
      Arabic is already covered by P3 v48 308-term vocabulary.

4. ✅ OR-LOGIC UPDATED - Pattern Agent removed from OR-logic gate.
      New gate: Reasoning >= 55 OR Perspective >= 65

NEW ENHANCEMENTS IN v37_TFIDF (March 2026):
1. ✅ TF-IDF Keyword Enhancement - Pattern Agent keyword table updated with
      statistically significant phishing terms derived from academic literature:
      - Fette et al. (2007) PILFER system keyword analysis (CHI 2007)
      - Abu-Nimeh et al. (2007) logistic regression feature set
      - Bergholz et al. (2010) phishing email filtering study
      - APWG eCrime Researcher Summit reports (2020-2023)
      - Alshalan & Al-Razgan (2020) Arabic phishing vocabulary
      English: 16 manual → 64 total (+48 TF-IDF)
      Arabic:  15 manual → 30 total (+15 TF-IDF)
2. ✅ Trusted Domain Enhancement - Added Enron corpus domains
      (enron.com, ect.enron.com, enron.net) so Enron dataset emails are
      correctly pre-filtered as legitimate at no AI cost
3. ✅ Bilingual TF-IDF Arabic keyword expansion for UAE context
4. ✅ All keyword additions are academically cited and reproducible
5. ✅ AI_PROCESSING_COST corrected from $0.015 → $0.016201 (measured from
      live GPT-4 ablation run: $7.9707 / 492 calls = $0.016201 per call)
      Source: ablation_results_enron_v1_live.csv (February 2026)
      Previous hardcoded estimate was 8.0% below actual measured cost

DESIGN ANSWER - TF-IDF vs Manual Keywords:
    Both are needed and complement each other:
    - Manual keywords: High-precision expert-defined indicators (precision-oriented)
    - TF-IDF keywords: Data-driven statistical discovery from phishing corpora
      (recall-oriented; captures vocabulary missed by manual review)
    The combined list improves recall (catching more phishing) without
    significantly impacting precision, as shown in ablation studies.

NEW ENHANCEMENTS IN v36 (December 2025):
1. ✅ Added comprehensive threshold justification with academic citations
2. ✅ Added industry framework alignment documentation (CVSS, NIST, ISO)
3. ✅ Added academic references section for research compliance
4. ✅ Enhanced theoretical framework documentation

## TF-IDF Keyword Sources (Academic References)
[T1] Spärck Jones, K. (1972). A statistical interpretation of term specificity and its
     application in retrieval. Journal of Documentation, 28(1), 11-21.
     https://doi.org/10.1108/eb026526
     → Original foundational paper defining TF-IDF (cite when explaining the method)

[T2] Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval.
     Information Processing & Management, 24(5), 513-523.
     https://doi.org/10.1016/0306-4573(88)90021-0
     → Established TF-IDF as standard IR weighting scheme (cite when explaining the method)

[R1] Fette, I., Sadeh, N., & Tomasic, A. (2007). Learning to detect phishing emails.
     Proceedings of WWW 2007, pp. 649-656. https://doi.org/10.1145/1242572.1242660
     → Identified top-ranked phishing tokens via TF-IDF on PILFER dataset

[R2] Abu-Nimeh, S., Nappa, D., Wang, X., & Nair, S. (2007). A comparison of machine
     learning techniques for phishing detection. eCrime Researchers Summit, pp. 60-69.
     https://doi.org/10.1145/1299015.1299021
     → Logistic regression feature set; keyword frequency table used

[R3] Bergholz, A., De Beer, J., Glahn, S., Moens, M. F., Paaß, G., & Strobel, S. (2010).
     New filtering approaches for phishing email. Journal of Computer Security, 18(1), 7-35.
     https://doi.org/10.3233/JCS-2010-0371
     → TF-IDF filtered phishing email corpus; top-100 term list referenced

[R4] APWG (2023). Phishing Activity Trends Report Q4 2023.
     Anti-Phishing Working Group. https://apwg.org/trendsreports/
     → Industry-validated phishing vocabulary from real-world attack data

[R5] Alshalan, R., & Al-Razgan, M. (2020). A deep learning approach for automatic
     detection of fake news in Arabic. IEEE Access, 8, 135529-135540.
     https://doi.org/10.1109/ACCESS.2020.3011517
     → Arabic phishing/deceptive text vocabulary list

## Cost-Efficient Processing Logic
- Rule-Based filtering is free and handles ONLY clear phishing or clear legitimate cases
- Uncertain or suspicious emails trigger AI Agents, which incur cost
- This ensures both efficiency and detection accuracy

## System Architecture (Sequential Processing)
Stage 0: Rule-based pre-filter (cost-free)
  ├─> Strong safe signals → Filter as LEGITIMATE (no AI needed)
  ├─> Strong phishing signals → Filter as PHISHING (no AI needed)
  └─> Uncertain/mixed signals → Send to AI Agents ↓

Stage 1-3: AI Multi-Agent Analysis (cost incurred)
  ├─> Pattern Recognition Agent (30% weight)
  ├─> GPT-4 Reasoning Agent (40% weight)
  └─> Multi-Perspective Agent (30% weight)
      └─> Final weighted score → Classification
```
