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

# ── configuration ──────────────────────────────────────────────────────────

TOP_K = 100

# Titles mapped to score contribution (title_match_score)
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
    "llm engineer",
}
GOOD_TITLES = {
    "senior data engineer", "data engineer", "backened engineer",
    "backend engineer", "software engineer", "senior software engineer",
    "full stack developer", "senior full stack developer",
    "analytics engineer", "data analyst",
    "senior data analyst", "cloud engineer",
}
NEUTRAL_TITLES = {
    "devops engineer", "senior devops engineer",
    "mobile developer", "frontend engineer", "qa engineer",
    "java developer", ".net developer",
}

# Industries
PRODUCT_INDUSTRIES = {
    "software", "fintech", "food delivery", "e-commerce",
    "ai/ml", "saas", "adtech", "healthtech", "gaming", "edtech",
    "conglomerate",  # Wayne Enterprises, Stark Industries etc.
}
CONSULTING_INDUSTRIES = {"it services", "consulting"}

# Consulting companies (penalize if entire career is here)
CONSULTING_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "mindtree", "hcl", "tech mahindra", "lti", "ltimindtree",
    "mphasis", "hexaware", "cyient", "persistent", "sasken",
    "robert bosch", "bosch",
}

# Product / high-quality companies (bonus)
PRODUCT_COMPANIES = {
    "razorpay", "zomato", "cred", "swiggy", "razorpay",
    "pied piper", "hooli", "stark industries", "wayne enterprises",
}

# Fictional companies treated as product cos
FICTIONAL_PRODUCT_COS = {"pied piper", "hooli", "stark industries", "wayne enterprises", "acme corp", "globex inc", "dunder mifflin", "initech"}

# Core AI/ML skill keywords relevant to the JD
CORE_AI_SKILLS = {
    "embeddings", "sentence-transformers", "bge", "e5",
    "retrieval", "ranking", "reranking", "hybrid search",
    "vector database", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "faiss",
    "rag", "retrieval augmented generation",
    "llm", "large language model", "fine-tuning", "lora", "qlora", "peft",
    "prompt engineering", "prompt",
    "transformers", "bert", "gpt", "t5",
    "ndcg", "mrr", "map", "evaluation", "a/b testing",
    "learning to rank", "xgboost", "gbdt",
    "information retrieval", "ir",
    "nlp", "natural language processing",
    "text classification", "ner", "text generation",
    "neural network", "deep learning",
    "langchain", "llamaindex",
    "python", "mlops", "feature engineering",
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

# Locations
INDIA_TIER_1_CITIES = {
    "pune", "noida", "bangalore", "hyderabad", "mumbai",
    "delhi", "gurgaon", "chennai", "kolkata", "ahmedabad",
}


def parse_date(date_str):
    """Parse YYYY-MM-DD to (year, month) tuple. Returns large tuple if None."""
    if not date_str or date_str == "null":
        return (9999, 1)
    parts = date_str.split("-")
    try:
        return (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return (9999, 1)


def days_between(d1, d2):
    """Approximate days between two dates (year, month) tuples."""
    return (d1[0] - d2[0]) * 365 + (d1[1] - d2[1]) * 30


def normalize_text(text):
    return re.sub(r"[^a-z0-9\s]", "", text.lower().strip())


def keyword_overlap_score(text, keywords):
    """Count how many keywords appear in text."""
    text_lower = normalize_text(text)
    count = 0
    for kw in keywords:
        if kw in text_lower:
            count += 1
    return count


def score_title(current_title):
    """Score the candidate's current title for JD alignment."""
    title_lower = current_title.lower().strip()

    # Direct match with ideal titles
    for t in IDEAL_TITLES:
        if t == title_lower or title_lower.startswith(t) or t.startswith(title_lower):
            return 30

    # Partial match: contains key terms
    if any(x in title_lower for x in ["ai", "machine learning", "ml", "nlp", "data scientist"]):
        if any(x in title_lower for x in ["engineer", "scientist", "specialist", "developer"]):
            return 28

    # Check for good titles
    for t in GOOD_TITLES:
        if t == title_lower or title_lower.startswith(t) or t.startswith(title_lower):
            return 18

    # NEUTRAL_TITLES
    for t in NEUTRAL_TITLES:
        if t == title_lower or title_lower.startswith(t) or t.startswith(title_lower):
            return 8

    # Contains "engineer" or "developer" but not matched above
    if "engineer" in title_lower or "developer" in title_lower:
        return 5

    # Technical-ish
    if any(x in title_lower for x in ["analyst", "architect", "technical", "scientist"]):
        return 4

    # Non-technical roles
    return 0


def score_experience(years):
    """Score years of experience. Ideal: 5-9."""
    if 5 <= years <= 9:
        return 15
    elif 4 <= years <= 10:
        return 12
    elif 3 <= years <= 11:
        return 8
    elif 2 <= years <= 12:
        return 5
    else:
        return 2


def score_skills(skills_list, career_histories, current_title=""):
    """
    Score skills relevance with anti-stuffing measures.
    Returns (score, relevant_skills_count, suspicious_flag).
    """
    if not skills_list:
        return (0, 0, False)

    score = 0
    core_count = 0
    total_skills = len(skills_list)

    # Look for keyword stuffing patterns
    expert_count = sum(1 for s in skills_list if s["proficiency"] == "expert")
    zero_duration_expert = sum(
        1 for s in skills_list
        if s["proficiency"] == "expert" and s.get("duration_months", 0) == 0
    )

    # Extract text from career descriptions for context
    career_text = " ".join(
        normalize_text(ch.get("description", ""))
        for ch in career_histories
    )

    for skill in skills_list:
        name = skill["name"].lower().strip()
        prof = skill["proficiency"]
        endorsements = skill.get("endorsements", 0)
        duration = skill.get("duration_months", 0)

        # Map proficiency to weight
        prof_weight = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.8, "expert": 1.0}.get(prof, 0.5)

        # Duration trust: if expert but 0 months used, heavily discount
        duration_trust = 1.0
        if prof == "expert" and duration == 0:
            duration_trust = 0.1
        elif prof == "expert" and duration < 6:
            duration_trust = 0.3
        elif prof == "advanced" and duration == 0:
            duration_trust = 0.3

        # Endorsement bonus (capped)
        end_bonus = min(endorsements / 100, 0.5)

        # Check if skill is core AI/ML
        is_core = False
        for kw in CORE_AI_SKILLS:
            if kw == name or name.startswith(kw) or kw.startswith(name):
                is_core = True
                break
        # Also check partial match
        if not is_core:
            for kw in CORE_AI_SKILLS:
                if kw in name or name in kw:
                    is_core = True
                    break

        if is_core:
            s = 3.0 * prof_weight * duration_trust + end_bonus
            score += s
            core_count += 1
        else:
            # Check if supporting skill
            is_supporting = False
            for kw in SUPPORTING_SKILLS:
                if kw == name or name.startswith(kw) or kw.startswith(name) or kw in name or name in kw:
                    is_supporting = True
                    break
            if is_supporting:
                score += 1.0 * prof_weight * duration_trust

    # Cap core skill bonus
    score = min(score, 30)

    # Detect keyword stuffer: many expert skills with 0 duration but not reflected in career
    suspicious = False
    if expert_count >= 5 and zero_duration_expert >= 3:
        suspicious = True
        score *= 0.3  # Heavy penalty
    elif expert_count >= 3 and zero_duration_expert >= 2:
        suspicious = True
        score *= 0.5

    # Another signal: too many core skills listed but no relevant career history
    if core_count >= 4:
        ai_terms_in_career = sum(
            1 for kw in ["machine learning", "ai", "ml", "nlp", "deep learning", "neural", "embedding", "rag", "llm", "data science"]
            if kw in career_text
        )
        if ai_terms_in_career < 2:
            score *= 0.35  # Skills don't match career
            suspicious = True

    # Check for skill diversity unrelated to career: expert in many domains but no focus
    title_lower = current_title.lower().strip()
    skill_domains = []
    ai_domain_kw = ["ai", "ml", "nlp", "llm", "neural", "embedding", "transformer", "rag", "retrieval", "ranking", "recommendation", "deep learning", "machine learning", "computer vision", "cv", "speech", "reinforcement", "rl", "gan"]
    infra_domain_kw = ["docker", "kubernetes", "aws", "gcp", "azure", "terraform", "jenkins", "ci/cd", "devops"]
    web_domain_kw = ["react", "angular", "vue", "node", "javascript", "typescript", "css", "html"]
    for s in skills_list:
        sn = s["name"].lower()
        if any(kw in sn for kw in ai_domain_kw):
            skill_domains.append("ai")
        elif any(kw in sn for kw in infra_domain_kw):
            skill_domains.append("infra")
        elif any(kw in sn for kw in web_domain_kw):
            skill_domains.append("web")
    unique_domains = set(skill_domains)
    if len(unique_domains) >= 3 and total_skills >= 10 and core_count >= 3:
        # Has AI + web + infra skills but in a non-AI role -> suspicious
        is_non_ai_role = (
            "engineer" not in title_lower
            or any(x in title_lower for x in ["marketing", "manager", "accountant", "sales", "hr", "support", "writer", "designer", "analyst"])
        )
        if is_non_ai_role:
            score *= 0.4
            suspicious = True

    return (score, core_count, suspicious)


def score_career_history(career_histories):
    """
    Evaluate career history for product-company experience, relevant roles, and anti-patterns.
    """
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

        # Company type
        if company in PRODUCT_COMPANIES or company in FICTIONAL_PRODUCT_COS:
            product_months += duration
            has_product_co = True
            all_consulting = False
            if duration >= 12:
                reasons.append(f"{ch['company']} (product, {duration}m)")
        elif company in CONSULTING_COMPANIES:
            consulting_months += duration
        elif industry in PRODUCT_INDUSTRIES:
            product_months += duration
            has_product_co = True
            all_consulting = False
            if duration >= 12:
                reasons.append(f"{ch['company']} (tech, {duration}m)")
        elif industry not in CONSULTING_INDUSTRIES:
            all_consulting = False

        # Check for relevant role (AI/ML/DS)
        title_lower = title
        for t in IDEAL_TITLES:
            if t in title_lower or title_lower in t:
                relevant_role_months += duration
                break
        else:
            if any(x in title_lower for x in ["data", "ai", "ml", "nlp", "machine learning", "analytics"]):
                if any(x in title_lower for x in ["engineer", "scientist", "analyst", "developer"]):
                    relevant_role_months += duration

    # Score
    score = 0

    # Product company experience is highly valued
    if has_product_co:
        score += 8
        if product_months >= 36:
            score += 4
        elif product_months >= 24:
            score += 2
    else:
        score += 0

    # Relevant role experience
    if relevant_role_months >= 36:
        score += 6
    elif relevant_role_months >= 24:
        score += 4
    elif relevant_role_months >= 12:
        score += 2

    # Penalty for all-consulting career
    if all_consulting and total_months > 24:
        score -= 5
        reasons.append("(consulting-only - penalty)")

    # Bonus for mixed (product + consulting)
    if product_months > 0 and consulting_months > 0:
        score += 2

    # Bonus for significant tenure at one company (>36 months)
    max_tenure = max(ch.get("duration_months", 0) for ch in career_histories)
    if max_tenure >= 36:
        score += 2  # Stability bonus

    # Check for production ML experience in descriptions
    prod_keywords = ["production", "deployed", "shipped", "launched", "serving", "inference", "pipeline"]
    desc_text = " ".join(normalize_text(ch.get("description", "")) for ch in career_histories)
    prod_match = sum(1 for kw in prod_keywords if kw in desc_text)
    if prod_match >= 2:
        score += 2

    score = max(-5, min(score, 20))
    return (score, "; ".join(reasons[:3]))


def score_education(education_list):
    """Score education relevance."""
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

        # Institution tier
        if tier == "tier_1":
            score += 2
        elif tier == "tier_2":
            score += 1

    return min(score, 8)


def score_location(location, country, willing_to_relocate, preferred_work_mode):
    """Score location fit for Pune/Noida-based role."""
    loc_lower = location.lower().strip()
    country_lower = country.lower().strip()

    # Already in Pune or Noida
    if "pune" in loc_lower:
        return (5, "Pune-based")
    if "noida" in loc_lower:
        return (5, "Noida-based")

    # Other Indian Tier-1 cities
    for city in INDIA_TIER_1_CITIES:
        if city in loc_lower:
            if willing_to_relocate:
                return (4, f"{location} (willing to relocate)")
            else:
                return (2, f"{location} (needs relocation)")

    # Other Indian city
    if "india" in country_lower or any(city in loc_lower for city in ["bhopal", "lucknow", "patna", "ranchi", "nagpur", "surat", "varanasi", "guwahati", "indore", "jaipur", "ahmedabad", "kochi", "trivandrum", "vizag", "coimbatore", "chandigarh", "bhubaneswar", "kolkata"]):
        if willing_to_relocate:
            return (3, f"{location} (willing to relocate)")
        else:
            return (1, f"{location} (India, needs relocation)")

    # Outside India
    return (0, f"{location} (outside India)")


def score_behavioral(signals):
    """Score behavioral/engagement signals."""
    score = 0
    reasons = []

    # Recruiter response rate (most important behavioral signal)
    resp_rate = signals.get("recruiter_response_rate", 0)
    if resp_rate >= 0.7:
        score += 5
        reasons.append(f"high response rate ({resp_rate:.0%})")
    elif resp_rate >= 0.4:
        score += 3
    elif resp_rate >= 0.2:
        score += 1

    # Open to work
    if signals.get("open_to_work_flag", False):
        score += 3
        reasons.append("open to work")

    # Notice period
    notice = signals.get("notice_period_days", 180)
    if notice <= 15:
        score += 3
        reasons.append(f"short notice ({notice}d)")
    elif notice <= 30:
        score += 2
        reasons.append(f"notice {notice}d")
    elif notice <= 60:
        score += 1
    elif notice > 90:
        score -= 2

    # Recent activity (last 30 days)
    # We approximate by checking if last_active_date is recent
    # The dataset is synthetic, so we use it as a signal
    if signals.get("search_appearance_30d", 0) > 0:
        score += 1
    if signals.get("saved_by_recruiters_30d", 0) > 0:
        score += 1

    # Willing to relocate bonus
    if signals.get("willing_to_relocate", False):
        score += 1

    # GitHub activity
    github = signals.get("github_activity_score", -1)
    if github >= 30:
        score += 1
    elif github == -1:
        pass  # No GitHub linked, not penalized

    # Interview completion rate (reliable candidates)
    interview_rate = signals.get("interview_completion_rate", 1.0)
    if interview_rate >= 0.9:
        score += 1
    elif interview_rate < 0.5:
        score -= 1

    # Offer acceptance rate
    offer_rate = signals.get("offer_acceptance_rate", -1)
    if offer_rate >= 0.8:
        score += 1
    elif offer_rate == -1:
        pass

    # Profile completeness
    completeness = signals.get("profile_completeness_score", 0)
    if completeness >= 90:
        score += 1

    # Salary expectation sanity check (for the role)
    salary = signals.get("expected_salary_range_inr_lpa", {})
    salary_min = salary.get("min", 0) if salary else 0
    salary_max = salary.get("max", 0) if salary else 0
    if 15 <= salary_min <= 60 or 15 <= salary_max <= 60:
        score += 1  # In range for Senior AI Engineer at Series A

    return (score, "; ".join(reasons[:3]))


def detect_honeypot(candidate):
    """
    Detect impossible-profile honeypots.
    Returns (is_honeypot, reason, confidence).
    """
    profile = candidate["profile"]
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    reasons = []
    confidence = 0

    # 1. Experience vs career history total
    total_career_months = sum(ch.get("duration_months", 0) for ch in career)
    total_career_years = total_career_months / 12
    stated_years = profile.get("years_of_experience", 0)

    years_diff = abs(stated_years - total_career_years)
    if years_diff > 3 and total_career_months > 0:
        reasons.append(f"YoE mismatch: stated {stated_years}y vs career {total_career_years:.1f}y")
        confidence += 3

    # 2. Expert in 8+ skills with 0 months duration
    expert_zero = [
        s["name"] for s in skills
        if s["proficiency"] == "expert" and s.get("duration_months", 0) == 0
    ]
    if len(expert_zero) >= 5:
        reasons.append(f"Expert in {len(expert_zero)} skills with 0mo experience")
        confidence += 4
    elif len(expert_zero) >= 3:
        reasons.append(f"Expert in {len(expert_zero)} skills with 0mo experience")
        confidence += 2

    # 3. Too many expert-level skills (10+) 
    expert_all = [s["name"] for s in skills if s["proficiency"] == "expert"]
    if len(expert_all) >= 10:
        reasons.append(f"{len(expert_all)} expert-level skills (suspicious)")
        confidence += 2

    # 4. Profile says AI/ML but career is completely unrelated
    title_lower = profile.get("current_title", "").lower()
    summary_lower = profile.get("summary", "").lower()
    career_text = " ".join(ch.get("description", "") for ch in career).lower()
    skills_text = " ".join(s["name"] for s in skills).lower()

    profile_has_ml = any(x in title_lower + summary_lower for x in ["ai", "machine learning", "ml", "deep learning", "neural", "nlp", "data science"])
    career_has_ml = any(x in career_text for x in ["machine learning", "ai ", " ml ", "deep learning", "neural", "nlp", "data science", "embedding", "rag", "llm"])

    if profile_has_ml and not career_has_ml and len(career) > 0:
        reasons.append("AI/ML profile but no ML experience in career history")
        confidence += 3

    # 5. Keyword stuffing: high AI skill count but irrelevant title
    title_is_nontech = score_title(profile.get("current_title", "")) <= 2
    ai_skills_count = sum(1 for s in skills if any(kw in s["name"].lower() for kw in ["ai", "ml", "nlp", "llm", "neural", "deep learning", "machine learning", "rag", "embedding", "transformer", "gpt", "bert"]))
    if title_is_nontech and ai_skills_count >= 5:
        reasons.append(f"Non-tech title ({profile['current_title']}) with {ai_skills_count} AI skills")
        confidence += 3

    # 6. All experience at a single company for unrealistically long
    if len(career) == 1:
        single_dur = career[0].get("duration_months", 0)
        if single_dur > 180:  # 15+ years at one company
            reasons.append(f"Single company for {single_dur//12}y")
            confidence += 1

    is_honeypot = confidence >= 4
    return (is_honeypot, "; ".join(reasons[:3]), confidence)


def score_candidate(candidate):
    """Score a single candidate and return a tuple of (score, reasoning, is_honeypot)."""
    profile = candidate["profile"]
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    cid = candidate["candidate_id"]

    # ── Honeypot check ──
    is_honeypot, hp_reason, hp_conf = detect_honeypot(candidate)

    # ── Score components ──
    title_score = score_title(profile.get("current_title", ""))
    exp_score = score_experience(profile.get("years_of_experience", 0))
    skills_score, core_count, suspicious_skills = score_skills(skills, career, profile.get("current_title", ""))
    career_score, career_reason = score_career_history(career)
    edu_score = score_education(education)
    loc_score, loc_reason = score_location(
        profile.get("location", ""),
        profile.get("country", ""),
        signals.get("willing_to_relocate", False),
        signals.get("preferred_work_mode", "flexible"),
    )
    behav_score, behav_reason = score_behavioral(signals)

    # ── Combine ──
    # Core fit (title + skills + career) dominates
    total = title_score + skills_score + career_score + exp_score + edu_score + loc_score + behav_score

    # Honeypot penalty
    if is_honeypot:
        total *= 0.01  # Effectively zero
        if hp_conf >= 6:
            total = -100  # Definitely disqualify

    # Build reasoning
    parts = []
    title = profile.get("current_title", "")
    yoe = profile.get("years_of_experience", 0)
    name = profile.get("anonymized_name", "")
    parts.append(f"{name} ({title}, {yoe}y)")

    if title_score >= 25:
        parts.append("title matches AI/ML role")
    elif title_score >= 15:
        parts.append("tech background")
    elif title_score <= 2:
        parts.append(f"non-ML role ({title})")

    if core_count > 0:
        parts.append(f"{core_count} AI-relevant skills")
    if career_reason:
        parts.append(career_reason[:80])
    if loc_reason:
        parts.append(loc_reason[:40])
    if behav_reason:
        parts.append(behav_reason[:60])

    if suspicious_skills:
        parts.append("(skill profile inconsistent)")

    if is_honeypot:
        parts.append(f"HONEYPOT: {hp_reason[:100]}")

    reasoning = "; ".join(parts)
    if len(reasoning) > 250:
        reasoning = reasoning[:247] + "..."

    new_score = max(total, -99)

    # If honeypot, still include but at very bottom
    if is_honeypot:
        new_score = -abs(new_score)

    return (new_score, reasoning, is_honeypot)


def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--max-candidates", type=int, default=None, help="Limit for testing")
    args = parser.parse_args()

    t0 = time.time()

    # ── Load candidates ──
    candidates = []
    print(f"Loading candidates from {args.candidates}...")
    with open(args.candidates, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    # Detect format: JSONL (one JSON per line) or JSON array
    if raw.startswith("["):
        candidates = json.loads(raw)
    else:
        for i, line in enumerate(raw.split("\n")):
            if line.strip():
                candidates.append(json.loads(line))
            if args.max_candidates and i + 1 >= args.max_candidates:
                break

    print(f"Loaded {len(candidates)} candidates in {time.time() - t0:.1f}s")

    # ── Score candidates ──
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

    # ── Sort and pick top 100 (non-honeypots first) ──
    # Sort by score descending, then by candidate_id for deterministic tiebreaking
    scored.sort(key=lambda x: (-x[0], x[1]))

    # Remove honeypots from the top
    non_hp = [s for s in scored if not s[3]]
    hp = [s for s in scored if s[3]]

    # Take top 100 from non-honeypots
    top_100 = non_hp[:TOP_K]

    # If not enough non-honeypots, fill with honeypots (shouldn't happen)
    if len(top_100) < TOP_K:
        top_100.extend(hp[: TOP_K - len(top_100)])

    # ── Normalize scores to 0-1 range ──
    max_score = top_100[0][0] if top_100 else 1
    min_score = top_100[-1][0] if top_100 else 0
    score_range = max_score - min_score if max_score != min_score else 1

    normalized = []
    for score, cid, reasoning, is_hp in top_100:
        norm = 0.2 + 0.8 * (score - min_score) / score_range  # Map to [0.2, 1.0]
        normalized.append((norm, cid, reasoning, is_hp))

    # Re-sort by normalized score descending (tiebreak by cid)
    normalized.sort(key=lambda x: (-x[0], x[1]))

    # ── Write CSV ──
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
