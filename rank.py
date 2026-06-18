#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Discovery & Ranking System

Scores 100K candidates against the Senior AI Engineer JD using:
  1. Role/title fit (anti-keyword-stuffer)
  2. Skills relevance with endorsement/duration trust
  3. Career-history signal (product cos > consulting)
  4. Location + relocation + notice period
  5. Behavioral engagement signals
  6. Honeypot / impossible-profile detection

Usage:
    python rank.py --candidates candidates.jsonl --out submission.csv
"""

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter

TOP_K = 100

# ── JD-derived signals ──────────────────────────────────────────────────────
# From the job description: Senior AI Engineer — Founding Team at Redrob AI
# Location: Pune/Noida (Hybrid)
# Experience: 5-9 years
# Key skills: embeddings, retrieval, ranking, LLMs, fine-tuning, vector DBs
# Anti-patterns: consulting-only, keyword stuffing, LangChain-only, pure research

IDEAL_TITLES = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "senior ai engineer", "senior ml engineer",
    "senior machine learning engineer", "senior software engineer (ml)",
    "ai specialist", "applied scientist", "research scientist",
    "senior data scientist", "data scientist",
    "senior ai specialist", "ml ops engineer",
    "deep learning engineer", "nlp engineer",
    "recommendation systems engineer", "recommendation engineer",
    "search engineer", "applied ml engineer",
    "machine learning scientist", "machine learning researcher",
    "senior machine learning scientist",
    "senior applied scientist", "senior research scientist",
    "ai ml engineer", "generative ai engineer",
    "llm engineer", "lead ai engineer", "staff machine learning engineer",
}
GOOD_TITLES = {
    "senior data engineer", "data engineer",
    "backend engineer", "software engineer", "senior software engineer",
    "full stack developer", "senior full stack developer",
    "analytics engineer", "data analyst",
    "senior data analyst", "cloud engineer",
    "senior backend engineer",
}
NEUTRAL_TITLES = {
    "devops engineer", "senior devops engineer",
    "mobile developer", "frontend engineer", "qa engineer",
    "java developer", ".net developer",
}

PRODUCT_INDUSTRIES = {
    "software", "fintech", "food delivery", "e-commerce",
    "ai/ml", "saas", "adtech", "healthtech", "gaming", "edtech",
    "conglomerate",
}
CONSULTING_INDUSTRIES = {"it services", "consulting"}

CONSULTING_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "mindtree", "hcl", "tech mahindra", "lti", "ltimindtree",
    "mphasis", "hexaware", "cyient", "persistent", "sasken",
}

PRODUCT_COMPANIES = {
    "razorpay", "zomato", "cred", "swiggy",
    "pied piper", "hooli", "stark industries", "wayne enterprises",
}

FICTIONAL_PRODUCT_COS = {
    "pied piper", "hooli", "stark industries", "wayne enterprises",
    "acme corp", "globex inc", "dunder mifflin", "initech",
}

# Core AI/ML skills from the JD
CORE_AI_SKILLS = {
    "embeddings", "sentence-transformers", "bge", "e5",
    "retrieval", "ranking", "reranking", "hybrid search",
    "vector database", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "faiss",
    "rag", "retrieval augmented generation",
    "llm", "large language model", "fine-tuning", "lora", "qlora", "peft",
    "prompt engineering",
    "transformers", "bert", "gpt", "t5",
    "ndcg", "mrr", "map", "evaluation", "a/b testing",
    "learning to rank", "xgboost", "gbdt",
    "information retrieval", "ir",
    "nlp", "natural language processing",
    "text classification", "ner", "text generation",
    "neural network", "deep learning",
    "langchain", "llamaindex",
    "python", "mlops", "feature engineering",
    "machine learning", "data science",
}

SUPPORTING_SKILLS = {
    "python", "sql", "pyspark", "spark", "airflow",
    "kubernetes", "docker", "aws", "gcp", "azure",
    "mlflow", "weights & biases", "wandb",
    "tensorflow", "pytorch", "jax", "scikit-learn",
    "pandas", "numpy", "scipy",
    "kafka", "flink", "beam", "data pipeline",
    "ci/cd", "git",
}

INDIA_TIER_1_CITIES = {
    "pune", "noida", "bangalore", "hyderabad", "mumbai",
    "delhi", "gurgaon", "chennai", "kolkata", "ahmedabad",
}


def normalize_text(text):
    return re.sub(r"[^a-z0-9\s]", "", text.lower().strip())


def score_title(current_title):
    title_lower = current_title.lower().strip()
    for t in IDEAL_TITLES:
        if t == title_lower or title_lower.startswith(t) or t.startswith(title_lower):
            return 30
    if any(x in title_lower for x in ["ai", "machine learning", "ml", "nlp", "data scientist"]):
        if any(x in title_lower for x in ["engineer", "scientist", "specialist", "developer"]):
            return 28
    for t in GOOD_TITLES:
        if t == title_lower or title_lower.startswith(t) or t.startswith(title_lower):
            return 18
    for t in NEUTRAL_TITLES:
        if t == title_lower or title_lower.startswith(t) or t.startswith(title_lower):
            return 8
    if "engineer" in title_lower or "developer" in title_lower:
        return 5
    if any(x in title_lower for x in ["analyst", "architect", "technical", "scientist"]):
        return 4
    return 0


def score_experience(years):
    if 5 <= years <= 9:
        return 15
    if 4 <= years <= 10:
        return 12
    if 3 <= years <= 11:
        return 8
    if 2 <= years <= 12:
        return 5
    return 2


def score_skills(skills_list, career_histories, current_title=""):
    if not skills_list:
        return (0, 0, False)

    score = 0
    core_count = 0
    total_skills = len(skills_list)

    expert_count = sum(1 for s in skills_list if s["proficiency"] == "expert")
    zero_duration_expert = sum(
        1 for s in skills_list
        if s["proficiency"] == "expert" and s.get("duration_months", 0) == 0
    )

    career_text = " ".join(normalize_text(ch.get("description", "")) for ch in career_histories)

    for skill in skills_list:
        name = skill["name"].lower().strip()
        prof = skill["proficiency"]
        endorsements = skill.get("endorsements", 0)
        duration = skill.get("duration_months", 0)

        prof_weight = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.8, "expert": 1.0}.get(prof, 0.5)

        duration_trust = 1.0
        if prof == "expert" and duration == 0:
            duration_trust = 0.15
        elif prof == "expert" and duration < 6:
            duration_trust = 0.35
        elif prof == "advanced" and duration == 0:
            duration_trust = 0.4

        end_bonus = min(endorsements / 150, 0.4)

        is_core = any(kw in name or name in kw for kw in CORE_AI_SKILLS)

        if is_core:
            s = 3.0 * prof_weight * duration_trust + end_bonus
            score += s
            core_count += 1
        else:
            is_supporting = any(kw in name or name in kw for kw in SUPPORTING_SKILLS)
            if is_supporting:
                score += 1.0 * prof_weight * duration_trust

    score = min(score, 30)

    suspicious = False

    # Keyword stuffing: expert in 5+ skills with 0 duration AND no ML career evidence
    if expert_count >= 5 and zero_duration_expert >= 3:
        ai_terms = sum(1 for kw in ["machine learning", "ai", "ml", "nlp", "deep learning", "neural", "embedding", "rag", "llm"] if kw in career_text)
        if ai_terms < 2:
            score *= 0.2
            suspicious = True

    # Core skills mismatch with career
    if core_count >= 4:
        ai_terms = sum(1 for kw in ["machine learning", "ai", "ml", "nlp", "deep learning", "neural", "embedding", "rag", "llm", "data science"] if kw in career_text)
        if ai_terms < 2:
            score *= 0.4
            suspicious = True

    # Non-tech title with AI skill spread
    title_lower = current_title.lower().strip()
    skill_domains = []
    ai_domain_kw = ["ai", "ml", "nlp", "llm", "neural", "embedding", "transformer", "rag", "retrieval", "ranking", "recommendation", "deep learning", "machine learning", "computer vision", "speech"]
    infra_domain_kw = ["docker", "kubernetes", "aws", "gcp", "azure", "terraform", "jenkins", "ci/cd"]
    web_domain_kw = ["react", "angular", "vue", "node", "javascript", "typescript", "css", "html"]
    for s in skills_list:
        sn = s["name"].lower()
        if any(kw in sn for kw in ai_domain_kw):
            skill_domains.append("ai")
        elif any(kw in sn for kw in infra_domain_kw):
            skill_domains.append("infra")
        elif any(kw in sn for kw in web_domain_kw):
            skill_domains.append("web")
    if len(set(skill_domains)) >= 3 and total_skills >= 10 and core_count >= 3:
        is_non_ai_role = (
            "engineer" not in title_lower
            or any(x in title_lower for x in ["marketing", "manager", "accountant", "sales", "hr", "support", "writer", "designer"])
        )
        if is_non_ai_role:
            score *= 0.35
            suspicious = True

    return (score, core_count, suspicious)


def score_career_history(career_histories):
    if not career_histories:
        return (0, "")

    total_months = sum(ch.get("duration_months", 0) for ch in career_histories)
    if total_months == 0:
        return (0, "")

    product_months = 0
    consulting_months = 0
    relevant_role_months = 0
    has_product_co = False
    all_consulting = True
    reasons = []

    for ch in career_histories:
        company = ch.get("company", "").lower().strip()
        title = ch.get("title", "").lower().strip()
        industry = ch.get("industry", "").lower().strip()
        duration = ch.get("duration_months", 0)
        description = normalize_text(ch.get("description", ""))

        if company in PRODUCT_COMPANIES or company in FICTIONAL_PRODUCT_COS:
            product_months += duration
            has_product_co = True
            all_consulting = False
            if duration >= 12:
                reasons.append(f"{ch['company']} ({duration}m)")
        elif company in CONSULTING_COMPANIES:
            consulting_months += duration
        elif industry in PRODUCT_INDUSTRIES:
            product_months += duration
            has_product_co = True
            all_consulting = False
            if duration >= 12:
                reasons.append(f"{ch['company']} ({duration}m)")
        elif industry not in CONSULTING_INDUSTRIES:
            all_consulting = False

        title_lower = title
        if any(x in title_lower for x in ["machine learning", "ml engineer", "ai engineer", "ai specialist", "nlp", "data scientist", "applied scientist", "research scientist", "recommendation", "search engineer"]):
            relevant_role_months += duration
        elif any(x in title_lower for x in ["data", "ai", "ml", "nlp"]):
            if any(x in title_lower for x in ["engineer", "scientist", "analyst"]):
                relevant_role_months += duration

    score = 4  # Base score

    if has_product_co:
        score += 6
        if product_months >= 36:
            score += 4
        elif product_months >= 24:
            score += 2

    if relevant_role_months >= 36:
        score += 6
    elif relevant_role_months >= 24:
        score += 4
    elif relevant_role_months >= 12:
        score += 2

    if all_consulting and total_months > 24:
        score -= 8
        reasons.append("(consulting-only)")
    elif all_consulting:
        score -= 4

    if product_months > 0 and consulting_months > 0:
        score += 2

    max_tenure = max(ch.get("duration_months", 0) for ch in career_histories)
    if max_tenure >= 36:
        score += 2

    prod_keywords = ["production", "deployed", "shipped", "launched", "serving", "inference"]
    desc_text = " ".join(normalize_text(ch.get("description", "")) for ch in career_histories)
    prod_match = sum(1 for kw in prod_keywords if kw in desc_text)
    if prod_match >= 2:
        score += 2

    score = max(-8, min(score, 22))
    return (score, "; ".join(reasons[:3]))


def score_education(education_list):
    if not education_list:
        return 2

    score = 0
    for edu in education_list:
        field = edu.get("field_of_study", "").lower()
        degree = edu.get("degree", "").lower()
        tier = edu.get("tier", "unknown")

        if "computer science" in field or "cs" in field:
            score += 4
        elif any(x in field for x in ["computer", "software", "engineering", "electrical", "electronics", "information", "data", "mathematics", "statistics", "physics"]):
            score += 3
        elif any(x in field for x in ["business", "management", "commerce", "arts"]):
            score += 1

        if "phd" in degree or "ph.d" in degree:
            score += 2
        elif "master" in degree or "m.tech" in degree or "ms" in degree:
            score += 1

        if tier == "tier_1":
            score += 2
        elif tier == "tier_2":
            score += 1

    return min(score, 8)


def score_location(location, country, willing_to_relocate):
    loc_lower = location.lower().strip()
    country_lower = country.lower().strip()

    if "pune" in loc_lower:
        return (6, "Pune-based")
    if "noida" in loc_lower:
        return (6, "Noida-based")

    for city in ["bangalore", "hyderabad", "mumbai", "delhi", "gurgaon", "chennai"]:
        if city in loc_lower:
            if willing_to_relocate:
                return (5, f"{location} (ready to relocate)")
            return (2, f"{location}")

    other_india = ["kolkata", "ahmedabad", "indore", "jaipur", "kochi", "trivandrum", "vizag", "coimbatore", "chandigarh", "bhubaneswar"]
    for city in other_india:
        if city in loc_lower:
            if willing_to_relocate:
                return (4, f"{location} (ready to relocate)")
            return (2, f"{location}")

    if "india" in country_lower:
        return (2, location)

    return (0, f"{location}")


def score_behavioral(signals):
    score = 0
    reasons = []

    resp_rate = signals.get("recruiter_response_rate", 0)
    if resp_rate >= 0.7:
        score += 5
        reasons.append(f"response rate {resp_rate:.0%}")
    elif resp_rate >= 0.4:
        score += 3
        reasons.append(f"response rate {resp_rate:.0%}")
    elif resp_rate >= 0.2:
        score += 1

    if signals.get("open_to_work_flag", False):
        score += 3
        reasons.append("open to work")

    notice = signals.get("notice_period_days", 180)
    if notice <= 15:
        score += 4
        reasons.append(f"notice {notice}d")
    elif notice <= 30:
        score += 3
        reasons.append(f"notice {notice}d")
    elif notice <= 60:
        score += 1
    elif notice > 90:
        score -= 2

    if signals.get("search_appearance_30d", 0) > 5:
        score += 1
    if signals.get("saved_by_recruiters_30d", 0) > 2:
        score += 1

    if signals.get("willing_to_relocate", False):
        score += 1

    github = signals.get("github_activity_score", -1)
    if github >= 30:
        score += 1

    interview_rate = signals.get("interview_completion_rate", 1.0)
    if interview_rate >= 0.9:
        score += 1
    elif interview_rate < 0.5:
        score -= 1

    offer_rate = signals.get("offer_acceptance_rate", -1)
    if offer_rate >= 0.8:
        score += 1

    completeness = signals.get("profile_completeness_score", 0)
    if completeness >= 90:
        score += 1

    salary = signals.get("expected_salary_range_inr_lpa", {}) or {}
    salary_min = salary.get("min", 0)
    salary_max = salary.get("max", 0)
    if 15 <= salary_min <= 60 or 15 <= salary_max <= 60:
        score += 1

    return (score, "; ".join(reasons[:3]))


def detect_honeypot(candidate):
    """
    Detect truly impossible profiles. Only flag clear-cut cases.
    Challenge says ~80 honeypots exist. We must be precise.
    """
    profile = candidate["profile"]
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    reasons = []
    strength = 0

    # Check 1: Years of experience completely mismatched with career duration
    total_career_months = sum(ch.get("duration_months", 0) for ch in career)
    total_career_years = total_career_months / 12
    stated_years = profile.get("years_of_experience", 0)

    if total_career_months > 0:
        diff = abs(stated_years - total_career_years)
        # Only flag if very large discrepancy (>5 years)
        if diff > 5 and total_career_years > 1:
            reasons.append(f"YoE mismatch: stated {stated_years}y, career sums to {total_career_years:.0f}y")
            strength += 5

    # Check 2: Expert in 8+ skills with ZERO months duration AND no career evidence
    expert_zero = [s for s in skills if s["proficiency"] == "expert" and s.get("duration_months", 0) == 0]
    if len(expert_zero) >= 8:
        reasons.append(f"Expert in {len(expert_zero)} skills with 0mo experience")
        strength += 5
    elif len(expert_zero) >= 5:
        # Also check if career text has any AI terms at all
        career_text = " ".join(ch.get("description", "") for ch in career).lower()
        ai_terms = sum(1 for kw in ["machine learning", "ai", "ml", "nlp", "python", "data", "algorithm", "model"] if kw in career_text)
        if ai_terms < 2:
            reasons.append(f"Expert in {len(expert_zero)} skills with 0mo, no career evidence")
            strength += 4

    # Check 3: Claims AI/ML title but career is entirely non-technical
    title_lower = profile.get("current_title", "").lower()
    is_ai_title = any(x in title_lower for x in ["ai engineer", "ml engineer", "machine learning", "data scientist", "nlp engineer", "ai specialist"])
    if is_ai_title:
        career_text = " ".join(ch.get("description", "") for ch in career).lower()
        has_tech_content = any(x in career_text for x in ["python", "machine learning", "model", "data", "algorithm", "code", "software", "engineer", "system", "pipeline", "api"])
        if not has_tech_content and len(career) > 0:
            reasons.append("AI title but no technical career history")
            strength += 4

    # Honeypot threshold: only flag if strength >= 5
    is_honeypot = strength >= 5
    return (is_honeypot, "; ".join(reasons[:2]), strength)


def score_candidate(candidate):
    profile = candidate["profile"]
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    cid = candidate["candidate_id"]

    is_honeypot, hp_reason, hp_strength = detect_honeypot(candidate)

    title_score = score_title(profile.get("current_title", ""))
    exp_score = score_experience(profile.get("years_of_experience", 0))
    skills_score, core_count, suspicious_skills = score_skills(skills, career, profile.get("current_title", ""))
    career_score, career_reason = score_career_history(career)
    edu_score = score_education(education)
    loc_score, loc_reason = score_location(
        profile.get("location", ""),
        profile.get("country", ""),
        signals.get("willing_to_relocate", False),
    )
    behav_score, behav_reason = score_behavioral(signals)

    # Weight: title + skills are primary signals (up to 60 combined)
    # Career + experience + education + location + behavioral are secondary
    total = title_score + skills_score + career_score + exp_score + edu_score + loc_score + behav_score

    # Heavily penalize honeypots
    if is_honeypot:
        total = -100 + (total * 0.01)

    # Build reasoning (max ~200 chars for clean output)
    parts = []
    title = profile.get("current_title", "")
    yoe = profile.get("years_of_experience", 0)
    name = profile.get("anonymized_name", "")

    parts.append(f"{title}, {yoe}y")

    if title_score >= 25:
        parts.append("AI/ML role match")
    elif title_score >= 15:
        parts.append("tech background")

    if core_count >= 5:
        parts.append(f"{core_count} core AI skills")
    elif core_count >= 2:
        parts.append(f"{core_count} AI skills")

    if career_reason:
        cr = career_reason[:80]
        parts.append(cr)

    if loc_reason:
        parts.append(loc_reason[:40])
    if behav_reason:
        parts.append(behav_reason[:60])
    if suspicious_skills:
        parts.append("(skill/career mismatch)")
    if is_honeypot:
        parts.append(f"[HONEYPOT]")

    reasoning = "; ".join(parts)
    if len(reasoning) > 250:
        reasoning = reasoning[:247] + "..."

    return (max(total, -99), reasoning, is_honeypot)


def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--max-candidates", type=int, default=None, help="Limit for testing")
    args = parser.parse_args()

    t0 = time.time()

    candidates = []
    print(f"Loading candidates from {args.candidates}...")
    with open(args.candidates, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if raw.startswith("["):
        candidates = json.loads(raw)
    else:
        for i, line in enumerate(raw.split("\n")):
            if line.strip():
                candidates.append(json.loads(line))
            if args.max_candidates and i + 1 >= args.max_candidates:
                break

    print(f"Loaded {len(candidates)} candidates in {time.time() - t0:.1f}s")

    t1 = time.time()
    scored = []
    honeypot_count = 0
    for c in candidates:
        score, reasoning, is_hp = score_candidate(c)
        scored.append((score, c["candidate_id"], reasoning, is_hp))
        if is_hp:
            honeypot_count += 1

    print(f"Scored {len(scored)} candidates in {time.time() - t1:.1f}s")
    print(f"Detected {honeypot_count} honeypots")

    scored.sort(key=lambda x: (-x[0], x[1]))

    # Separate honeypots; only include if we don't have enough non-honeypots
    non_hp = [s for s in scored if not s[3]]
    hp = [s for s in scored if s[3]]
    top_100 = non_hp[:TOP_K]
    if len(top_100) < TOP_K:
        top_100.extend(hp[: TOP_K - len(top_100)])

    # Normalize scores to [0.2, 1.0]
    max_score = top_100[0][0] if top_100 else 1
    min_score = top_100[-1][0] if top_100 else 0
    score_range = max_score - min_score if max_score != min_score else 1
    normalized = []
    for score, cid, reasoning, is_hp in top_100:
        norm = 0.2 + 0.8 * (score - min_score) / score_range
        normalized.append((norm, cid, reasoning, is_hp))
    normalized.sort(key=lambda x: (-x[0], x[1]))

    t2 = time.time()
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (score, cid, reasoning, _) in enumerate(normalized, start=1):
            writer.writerow([cid, rank, f"{score:.4f}", reasoning])

    print(f"Wrote {args.out} in {time.time() - t2:.1f}s")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
