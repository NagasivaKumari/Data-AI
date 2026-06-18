# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

A rule-based candidate ranking system for the Senior AI Engineer — Founding Team role at Redrob AI.

## Approach

Five-component scoring with behavioral modifier:

1. **Title/Role Fit (30 pts)** — Matches current title against the JD's ideal profile (AI/ML Engineer, Recommendation Systems Engineer, NLP Engineer, etc.)
2. **Skills Relevance (30 pts)** — Weights AI/ML core skills (embeddings, retrieval, ranking, LLMs, vector DBs, etc.) by proficiency, duration, and endorsements. Keyword-stuffer detection: expert-level skills with 0 months duration are heavily discounted; cross-domain skill spread in non-technical roles is penalized.
3. **Career History (20 pts)** — Rewards product-company experience (Razorpay, CRED, Swiggy, Zomato, etc.) vs. consulting-only backgrounds (TCS, Infosys, Wipro). Checks career descriptions for production ML evidence.
4. **Location & Logistics (5 pts)** — Pune/Noida preferred; other Indian Tier-1 cities with relocation willingness score well.
5. **Behavioral Signals (10 pts)** — Recruiter response rate, open-to-work flag, notice period, recent activity, interview completion rate, salary expectation alignment.

**Honeypot detection**: Flags impossible profiles via YoE vs. career-total mismatch, expert skills with zero duration, AI/ML profile without matching career history, and non-technical titles with heavy AI skill listings.

## Scoring formula

```
total = title_score + skills_score + career_score + exp_score + edu_score + loc_score + behav_score
```

If honeypot detected → score is zeroed. Scores are min-max normalized to [0.2, 1.0] for the output CSV.

## Reproduction

### Prerequisites
- Python 3.10+
- No external ML dependencies (pure Python standard library)
- 16 GB RAM, CPU only

### Command
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

### Runtime
~75 seconds for 100K candidates on a single CPU core. Well within the 5-minute limit.

### Pre-computation
None required. The ranker processes candidates on-the-fly from the JSONL file.

## Files

| File | Description |
|------|-------------|
| `rank.py` | Main ranking script |
| `submission.csv` | Output: top 100 ranked candidates |
| `submission_metadata.yaml` | Submission metadata |
| `requirements.txt` | Python dependencies (stdlib only) |

## Sandbox

The ranker accepts any JSON or JSONL candidate file and produces the submission CSV. For testing on a small sample (≤100 candidates), use the included `sample_candidates.json`.

## AI Tools Declaration

AI tools (Claude) were used for architecture discussion, code review, and debugging. No candidate data was fed to any LLM. The ranking logic is entirely rule-based and runs locally on CPU.
