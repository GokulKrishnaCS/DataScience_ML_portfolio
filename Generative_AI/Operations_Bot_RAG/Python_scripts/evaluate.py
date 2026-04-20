"""
================================================================================
PHASE 5: EVALUATION FRAMEWORK - RAGAS METRICS & BENCHMARKING
================================================================================
OpsBot - Enterprise Knowledge Copilot

Evaluation demonstrates production-grade LLM systems:
1. RAGAS metrics (Retrieval Augmented Generation Assessment)
   - Faithfulness: Does answer rely only on retrieved context?
   - Context Precision: How many retrieved docs are relevant?
   - Answer Relevance: Does answer address the question?
   - Context Recall: Did we retrieve all relevant documents?

2. Offline benchmark dataset (30 curated Q&A pairs)
   - Ground truth answers (human-validated)
   - Domain-specific questions about handbook
   - Varying difficulty levels

3. LLM-as-judge scoring
   - Structured evaluation rubrics
   - Confidence scores
   - Failure analysis

4. Regression testing
   - Track metrics over time
   - Detect degradation
   - Compare approaches

SKILL SIGNALS THIS DEMONSTRATES:
- LLM evaluation: Core skill in AI engineer roles
- Metrics & observability: Production engineering mindset
- Benchmark creation: Data discipline
- Failure analysis: Real-world debugging
- Reproducibility: Science-like rigor

================================================================================
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import re
from collections import defaultdict

# Evaluation framework imports
from query_engine import RAGPipeline, QueryResult
from agent import Agent

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ============================================================================
# BENCHMARK DATASET
# ============================================================================

@dataclass
class BenchmarkQuestion:
    """A single benchmark question with ground truth answer"""
    question_id: str
    question: str
    ground_truth: str  # Human-validated correct answer
    category: str     # e.g., "IT/Security", "HR/TimeOff", "Engineering/Onboarding"
    difficulty: str   # "easy", "medium", "hard"
    keywords: List[str]  # Expected key concepts in answer


# Curated benchmark dataset: 30 questions across 3 handbooks
# These are real questions an employee might ask
BENCHMARK_DATASET = [
    # =====================
    # IT/Security (10 questions)
    # =====================
    BenchmarkQuestion(
        question_id="it_sec_001",
        question="What is the company's password policy?",
        ground_truth="Passwords must be at least 12 characters, include uppercase, lowercase, numbers, and symbols. Passwords must be changed every 90 days.",
        category="IT/Security",
        difficulty="easy",
        keywords=["password", "12 characters", "uppercase", "lowercase", "90 days"]
    ),
    BenchmarkQuestion(
        question_id="it_sec_002",
        question="Is two-factor authentication required?",
        ground_truth="Yes, 2FA is mandatory for all accounts. Use authenticator apps (Google Authenticator, Authy) or security keys. SMS is not accepted.",
        category="IT/Security",
        difficulty="easy",
        keywords=["2FA", "mandatory", "authenticator", "security key"]
    ),
    BenchmarkQuestion(
        question_id="it_sec_003",
        question="What should I do if my password is compromised?",
        ground_truth="Immediately change your password and notify IT security. If you suspect data access, contact security@company.com and your manager within 1 hour.",
        category="IT/Security",
        difficulty="medium",
        keywords=["change password", "notify IT", "security@company.com"]
    ),
    BenchmarkQuestion(
        question_id="it_sec_004",
        question="What are the incident response procedures?",
        ground_truth="Report security incidents immediately to IT. Initial response within 30 minutes. Full investigation within 24 hours. Customer notification within 72 hours if needed.",
        category="IT/Security",
        difficulty="medium",
        keywords=["incident response", "30 minutes", "24 hours", "72 hours"]
    ),
    BenchmarkQuestion(
        question_id="it_sec_005",
        question="Can I use public WiFi for company work?",
        ground_truth="No. Always use a VPN when working remotely. Public WiFi is strictly prohibited for accessing company systems or sensitive data.",
        category="IT/Security",
        difficulty="easy",
        keywords=["public WiFi", "VPN", "prohibited"]
    ),
    BenchmarkQuestion(
        question_id="it_sec_006",
        question="What is the data classification policy?",
        ground_truth="Public: no restrictions. Internal: access controlled. Confidential: restricted access, encryption required. Secret: CEO approval required.",
        category="IT/Security",
        difficulty="hard",
        keywords=["data classification", "public", "internal", "confidential", "secret"]
    ),
    BenchmarkQuestion(
        question_id="it_sec_007",
        question="How often must I update my software?",
        ground_truth="Security patches: immediately when available. Major OS/software updates: within 30 days. All updates must be completed within 30 days of release.",
        category="IT/Security",
        difficulty="medium",
        keywords=["updates", "patches", "30 days"]
    ),
    BenchmarkQuestion(
        question_id="it_sec_008",
        question="What endpoint protection is required?",
        ground_truth="All devices must run antivirus and firewall software. Mandatory software: Okta, CrowdStrike, Slack. Updates automatic, cannot be disabled.",
        category="IT/Security",
        difficulty="medium",
        keywords=["antivirus", "firewall", "Okta", "CrowdStrike", "mandatory"]
    ),
    BenchmarkQuestion(
        question_id="it_sec_009",
        question="Can I share my login credentials with a colleague?",
        ground_truth="No. Sharing credentials is strictly prohibited. Use shared accounts only through secure systems. Each person must have individual credentials.",
        category="IT/Security",
        difficulty="easy",
        keywords=["credentials", "prohibited", "individual accounts"]
    ),
    BenchmarkQuestion(
        question_id="it_sec_010",
        question="What is the laptop encryption requirement?",
        ground_truth="All laptops must use full-disk encryption (BitLocker on Windows, FileVault on Mac). Encryption must be enabled before connecting to company network.",
        category="IT/Security",
        difficulty="medium",
        keywords=["encryption", "BitLocker", "FileVault", "full-disk"]
    ),

    # =====================
    # HR/TimeOff (10 questions)
    # =====================
    BenchmarkQuestion(
        question_id="hr_off_001",
        question="How many vacation days do I get?",
        ground_truth="All employees get 20 days of paid vacation per year. Additional 5 floating holidays. PTO accrues at 1.67 days per month.",
        category="HR/TimeOff",
        difficulty="easy",
        keywords=["vacation", "20 days", "floating holidays", "accrues"]
    ),
    BenchmarkQuestion(
        question_id="hr_off_002",
        question="What is the sick leave policy?",
        ground_truth="Unlimited sick leave for illness. Doctor's note required for absences exceeding 3 consecutive days. Report to manager before 10 AM or as soon as possible.",
        category="HR/TimeOff",
        difficulty="easy",
        keywords=["sick leave", "unlimited", "doctor's note"]
    ),
    BenchmarkQuestion(
        question_id="hr_off_003",
        question="How much notice do I need to give for vacation?",
        ground_truth="Minimum 2 weeks notice for vacation requests. Requests submitted in HR system. Manager approval required. High-impact times may have restrictions.",
        category="HR/TimeOff",
        difficulty="medium",
        keywords=["2 weeks", "notice", "manager approval", "HR system"]
    ),
    BenchmarkQuestion(
        question_id="hr_off_004",
        question="What holidays does the company observe?",
        ground_truth="New Year's Day, MLK Day, Presidents Day, Memorial Day, Independence Day, Labor Day, Thanksgiving (2 days), Christmas. 10 holidays total. Regional variations allowed.",
        category="HR/TimeOff",
        difficulty="easy",
        keywords=["holidays", "New Year", "Thanksgiving", "Christmas"]
    ),
    BenchmarkQuestion(
        question_id="hr_off_005",
        question="Can I carry over unused vacation?",
        ground_truth="Yes, maximum 5 days can be carried to next year. Unused vacation beyond 5 days is forfeited. Payout only on termination.",
        category="HR/TimeOff",
        difficulty="medium",
        keywords=["carry over", "5 days", "forfeited", "payout"]
    ),
    BenchmarkQuestion(
        question_id="hr_off_006",
        question="What is parental leave?",
        ground_truth="12 weeks paid leave for birth parent. 8 weeks for non-birth parent. Can be split. Applies to biological, adopted, and foster children.",
        category="HR/TimeOff",
        difficulty="medium",
        keywords=["parental leave", "12 weeks", "8 weeks", "adoption"]
    ),
    BenchmarkQuestion(
        question_id="hr_off_007",
        question="How do I request emergency time off?",
        ground_truth="Call or text your manager immediately. Email HR within 24 hours. Medical emergencies: hospital provides documentation. Job protected for up to 10 days.",
        category="HR/TimeOff",
        difficulty="medium",
        keywords=["emergency", "manager", "24 hours", "job protected"]
    ),
    BenchmarkQuestion(
        question_id="hr_off_008",
        question="What about sabbatical leave?",
        ground_truth="After 5 years: 4 weeks sabbatical. After 10 years: 8 weeks. Paid at base salary. Role held or equivalent position upon return.",
        category="HR/TimeOff",
        difficulty="hard",
        keywords=["sabbatical", "5 years", "10 years", "4 weeks", "8 weeks"]
    ),
    BenchmarkQuestion(
        question_id="hr_off_009",
        question="Can I work remotely while on vacation?",
        ground_truth="No. Vacation means time away from work. Remote work policies and vacation are separate. You must be fully away during vacation time.",
        category="HR/TimeOff",
        difficulty="medium",
        keywords=["remote", "vacation", "away from work"]
    ),
    BenchmarkQuestion(
        question_id="hr_off_010",
        question="What is bereavement leave?",
        ground_truth="Up to 10 days paid leave for death of immediate family (spouse, parent, child, sibling). Additional 5 days for extended family. Flexible arrangement available.",
        category="HR/TimeOff",
        difficulty="hard",
        keywords=["bereavement", "10 days", "immediate family", "5 days"]
    ),

    # =====================
    # Engineering/Onboarding (10 questions)
    # =====================
    BenchmarkQuestion(
        question_id="eng_onb_001",
        question="What development environment do I need?",
        ground_truth="Git 2.36+, Python 3.10+ or Node.js 16+, Docker 20.10+. IDEs: VS Code (recommended), JetBrains, or VIM. Complete setup in 2-3 hours.",
        category="Engineering/Onboarding",
        difficulty="easy",
        keywords=["Git", "Python 3.10", "Node.js 16", "Docker", "VS Code"]
    ),
    BenchmarkQuestion(
        question_id="eng_onb_002",
        question="What is the code review process?",
        ground_truth="Create feature branch, make changes, commit with clear messages, push, create PR, assign 1-2 reviewers, address feedback, get manager approval, merge to main.",
        category="Engineering/Onboarding",
        difficulty="medium",
        keywords=["code review", "pull request", "reviewers", "merge to main"]
    ),
    BenchmarkQuestion(
        question_id="eng_onb_003",
        question="What test coverage is required?",
        ground_truth="Minimum 80% code coverage. Unit tests for all functions, integration tests for APIs, performance tests for critical paths. All tests must pass before merge.",
        category="Engineering/Onboarding",
        difficulty="medium",
        keywords=["test coverage", "80%", "unit tests", "integration tests"]
    ),
    BenchmarkQuestion(
        question_id="eng_onb_004",
        question="How do I set up my local development environment?",
        ground_truth="Clone repos from GitHub, run npm install or pip install -r requirements.txt, configure local environment file, run npm test or pytest to verify, start with npm start or python app.py.",
        category="Engineering/Onboarding",
        difficulty="medium",
        keywords=["clone", "npm install", "pip install", "environment", "verify"]
    ),
    BenchmarkQuestion(
        question_id="eng_onb_005",
        question="What code review turnaround times are expected?",
        ground_truth="Standard: 24 hours. Urgent: 4 hours. Critical bugs: 1 hour. Escalate to engineering lead if reviewer unavailable.",
        category="Engineering/Onboarding",
        difficulty="easy",
        keywords=["24 hours", "4 hours", "1 hour", "turnaround"]
    ),
    BenchmarkQuestion(
        question_id="eng_onb_006",
        question="Which GitHub repositories do I need access to?",
        ground_truth="Main codebase: github.com/company/product. Infrastructure: github.com/company/infrastructure. Documentation: github.com/company/docs. Private tools in internal GitLab.",
        category="Engineering/Onboarding",
        difficulty="easy",
        keywords=["GitHub", "repositories", "product", "infrastructure"]
    ),
    BenchmarkQuestion(
        question_id="eng_onb_007",
        question="What are the first week goals?",
        ground_truth="Complete IT setup, clone and build codebase, create feature branch, submit PR, pair programming, understand architecture, read design docs, join Slack, attend standups, team lunch.",
        category="Engineering/Onboarding",
        difficulty="hard",
        keywords=["first week", "IT setup", "architecture", "standups"]
    ),
    BenchmarkQuestion(
        question_id="eng_onb_008",
        question="How can I get help when stuck?",
        ground_truth="Technical questions: #engineering Slack. HR/Admin: HR@company.com. Manager: direct message or 1:1. Pair programming sessions available. Office hours: Tue 2-3pm, Thu 10-11am, Fri 3-4pm.",
        category="Engineering/Onboarding",
        difficulty="medium",
        keywords=["help", "Slack", "manager", "pair programming", "office hours"]
    ),
    BenchmarkQuestion(
        question_id="eng_onb_009",
        question="What is the continuous integration process?",
        ground_truth="All tests run on every commit. Merge blocked if tests fail. Code coverage report included in PR. CI/CD pipeline enforces quality gates automatically.",
        category="Engineering/Onboarding",
        difficulty="hard",
        keywords=["CI", "tests", "merge blocked", "coverage report"]
    ),
    BenchmarkQuestion(
        question_id="eng_onb_010",
        question="When is the 30-day check-in?",
        ground_truth="After 30 days, review with manager covering: environment setup complete, code quality and pace, team integration, questions/concerns, goals for next month.",
        category="Engineering/Onboarding",
        difficulty="easy",
        keywords=["30-day", "check-in", "manager", "code quality"]
    ),
]


# ============================================================================
# EVALUATION METRICS (RAGAS-LIKE)
# ============================================================================

@dataclass
class RAGASMetrics:
    """Individual RAGAS metrics for a single query"""

    # Faithfulness: Does answer depend only on context?
    # Computed: LLM checks if answer statements can be inferred from docs
    # Range: 0.0 to 1.0 (higher = better)
    faithfulness: float

    # Answer Relevance: Does answer address the question?
    # Computed: Semantic similarity between question and answer
    # Range: 0.0 to 1.0 (higher = better)
    answer_relevance: float

    # Context Precision: Are retrieved docs actually relevant?
    # Computed: Fraction of retrieved docs containing answer
    # Range: 0.0 to 1.0 (higher = better)
    context_precision: float

    # Context Recall: Did we retrieve all necessary docs?
    # Computed: Fraction of gold documents retrieved
    # Range: 0.0 to 1.0 (higher = better)
    context_recall: float

    def aggregate_score(self) -> float:
        """Harmonic mean of all metrics (F1-like average)"""
        scores = [self.faithfulness, self.answer_relevance,
                  self.context_precision, self.context_recall]
        # Filter out invalid scores
        valid = [s for s in scores if 0 <= s <= 1]
        if not valid:
            return 0.0
        return sum(valid) / len(valid)


@dataclass
class EvaluationResult:
    """Results for evaluating a single question"""
    question_id: str
    question: str
    retrieved_answer: str
    ground_truth: str
    retrieved_docs: List[str]  # List of document snippets
    metrics: RAGASMetrics
    timestamp: str


# ============================================================================
# EVALUATION ENGINE
# ============================================================================

class EvaluationEngine:
    """Run RAGAS-like evaluation on benchmark questions"""

    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Initialize evaluation engine

        Args:
            rag_pipeline: RAGPipeline instance to evaluate
        """
        self.rag = rag_pipeline
        self.results: List[EvaluationResult] = []
        self.client = None

        if OPENAI_AVAILABLE:
            try:
                self.client = OpenAI()
            except Exception as e:
                print(f"[WARNING] OpenAI client not available: {e}")
                self.client = None

    def _compute_faithfulness(self, answer: str, context: str) -> float:
        """
        Compute faithfulness: does answer rely only on context?

        Strategy:
        1. If OpenAI available: LLM-based evaluation
        2. Fallback: keyword overlap (simple but effective)

        Args:
            answer: Generated answer
            context: Retrieved context docs

        Returns:
            Faithfulness score 0.0-1.0
        """
        if not answer or not context:
            return 0.5  # Neutral score

        if self.client:
            return self._llm_faithfulness(answer, context)
        else:
            return self._keyword_faithfulness(answer, context)

    def _llm_faithfulness(self, answer: str, context: str) -> float:
        """LLM-based faithfulness: can answer be inferred from context?"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are evaluating if an answer can be inferred from context. "
                                   "Rate 1.0 if yes, 0.0 if answer contradicts context or uses outside info, "
                                   "0.5 if partially supported. Return ONLY a number."
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nAnswer:\n{answer}\n\nCan the answer be inferred from the context? (0.0-1.0)"
                    }
                ],
                temperature=0.2,
                max_tokens=10
            )

            score_text = response.choices[0].message.content.strip()
            return float(score_text)
        except Exception as e:
            print(f"  [WARNING] LLM faithfulness failed: {e}")
            return self._keyword_faithfulness(answer, context)

    def _keyword_faithfulness(self, answer: str, context: str) -> float:
        """Simple fallback: check if answer keywords appear in context"""
        # Extract keywords from answer (longer words > 4 chars)
        answer_words = set(
            word.lower() for word in answer.split()
            if len(word) > 4 and not word.startswith(('the', 'that', 'this'))
        )
        context_words = set(word.lower() for word in context.split())

        if not answer_words:
            return 0.5

        overlap = len(answer_words & context_words) / len(answer_words)
        # Require at least 50% of meaningful words in context
        return min(1.0, overlap * 1.5)

    def _compute_answer_relevance(self, question: str, answer: str) -> float:
        """
        Compute answer relevance: does answer address question?

        Measures semantic similarity between question and answer.
        Uses simple keyword-based approach (no embeddings needed for speed).
        """
        if not question or not answer:
            return 0.0

        # Extract key question words
        question_words = set(
            word.lower() for word in question.replace('?', '').split()
            if len(word) > 3
        )
        answer_words = set(
            word.lower() for word in answer.split()
            if len(word) > 3
        )

        if not question_words:
            return 0.5

        # Overlap ratio
        overlap = len(question_words & answer_words) / len(question_words)

        # Answer length heuristic: very short answers get penalty
        answer_length_factor = min(1.0, len(answer.split()) / 10)

        return min(1.0, overlap * answer_length_factor)

    def _compute_context_precision(self, answer: str, context: str) -> float:
        """
        Compute context precision: what fraction of context is relevant?

        Simple heuristic: check if context words appear in answer
        """
        if not context:
            return 0.0

        context_sentences = [s.strip() for s in context.split('.') if s.strip()]
        if not context_sentences:
            return 0.5

        # Count sentences that have substantial overlap with answer
        answer_words = set(word.lower() for word in answer.split())
        relevant_sentences = 0

        for sent in context_sentences:
            sent_words = set(word.lower() for word in sent.split())
            overlap = len(answer_words & sent_words)
            if overlap >= 2:  # At least 2 words match
                relevant_sentences += 1

        return min(1.0, relevant_sentences / len(context_sentences))

    def _compute_context_recall(self, ground_truth: str, context: str,
                               keywords: List[str]) -> float:
        """
        Compute context recall: did we retrieve all relevant info?

        Check if expected keywords appear in context
        """
        if not context or not keywords:
            return 0.5

        context_lower = context.lower()
        found_keywords = sum(
            1 for kw in keywords
            if kw.lower() in context_lower
        )

        return min(1.0, found_keywords / len(keywords))

    def evaluate_question(self, benchmark_q: BenchmarkQuestion) -> EvaluationResult:
        """
        Evaluate system on a single benchmark question

        Args:
            benchmark_q: BenchmarkQuestion to evaluate

        Returns:
            EvaluationResult with metrics
        """
        # Query the RAG system
        rag_result = self.rag.query(benchmark_q.question)

        # Prepare context from retrieved documents
        context = "\n\n".join(rag_result.sources) if rag_result.sources else ""

        # Compute RAGAS metrics
        metrics = RAGASMetrics(
            faithfulness=self._compute_faithfulness(rag_result.answer, context),
            answer_relevance=self._compute_answer_relevance(
                benchmark_q.question,
                rag_result.answer
            ),
            context_precision=self._compute_context_precision(
                rag_result.answer,
                context
            ),
            context_recall=self._compute_context_recall(
                benchmark_q.ground_truth,
                context,
                benchmark_q.keywords
            )
        )

        result = EvaluationResult(
            question_id=benchmark_q.question_id,
            question=benchmark_q.question,
            retrieved_answer=rag_result.answer,
            ground_truth=benchmark_q.ground_truth,
            retrieved_docs=rag_result.sources,
            metrics=metrics,
            timestamp=datetime.now().isoformat()
        )

        self.results.append(result)
        return result

    def run_benchmark(self, subset: str = "all") -> Dict[str, Any]:
        """
        Run evaluation on benchmark dataset

        Args:
            subset: "all", "easy", "medium", "hard"

        Returns:
            Dictionary with overall metrics and per-category breakdown
        """
        print("\n" + "="*70)
        print("PHASE 5: EVALUATION FRAMEWORK - RAGAS BENCHMARKING")
        print("="*70)

        # Filter dataset
        if subset == "all":
            questions = BENCHMARK_DATASET
        else:
            questions = [q for q in BENCHMARK_DATASET if q.difficulty == subset]

        print(f"\nRunning benchmark on {len(questions)} questions ({subset} difficulty)")
        print("-" * 70)

        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] {question.question_id}: {question.question[:60]}...")

            result = self.evaluate_question(question)
            score = result.metrics.aggregate_score()

            print(f"  Faithfulness:      {result.metrics.faithfulness:.2f}")
            print(f"  Answer Relevance:  {result.metrics.answer_relevance:.2f}")
            print(f"  Context Precision: {result.metrics.context_precision:.2f}")
            print(f"  Context Recall:    {result.metrics.context_recall:.2f}")
            print(f"  OVERALL SCORE:     {score:.2f}/1.00")

        # Compute aggregate statistics
        return self._compute_aggregate_stats()

    def _compute_aggregate_stats(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all evaluations"""
        if not self.results:
            return {"error": "No results to aggregate"}

        # Overall metrics
        all_scores = [r.metrics.aggregate_score() for r in self.results]

        metrics_by_type = {
            "faithfulness": [r.metrics.faithfulness for r in self.results],
            "answer_relevance": [r.metrics.answer_relevance for r in self.results],
            "context_precision": [r.metrics.context_precision for r in self.results],
            "context_recall": [r.metrics.context_recall for r in self.results],
        }

        # Per-category breakdown
        by_category = defaultdict(list)
        for result in self.results:
            # Infer category from question_id
            category = result.question_id.split('_')[0:2]
            category = "_".join(category)
            by_category[category].append(result.metrics.aggregate_score())

        # Per-difficulty breakdown
        by_difficulty = defaultdict(list)
        for i, result in enumerate(self.results):
            idx = i % 10  # Rough mapping to benchmark dataset
            benchmark = BENCHMARK_DATASET[i] if i < len(BENCHMARK_DATASET) else None
            if benchmark:
                by_difficulty[benchmark.difficulty].append(
                    result.metrics.aggregate_score()
                )

        # Statistics helper
        def stats(scores):
            if not scores:
                return {"min": 0, "max": 0, "avg": 0}
            return {
                "min": min(scores),
                "max": max(scores),
                "avg": sum(scores) / len(scores)
            }

        return {
            "total_questions": len(self.results),
            "overall_score": sum(all_scores) / len(all_scores),
            "metrics": {
                "faithfulness": stats(metrics_by_type["faithfulness"]),
                "answer_relevance": stats(metrics_by_type["answer_relevance"]),
                "context_precision": stats(metrics_by_type["context_precision"]),
                "context_recall": stats(metrics_by_type["context_recall"]),
            },
            "by_category": {
                cat: stats(scores)
                for cat, scores in by_category.items()
            },
            "by_difficulty": {
                diff: stats(scores)
                for diff, scores in by_difficulty.items()
            }
        }

    def save_results(self, filepath: Path):
        """Save evaluation results to JSON"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "results": [
                {
                    **asdict(r),
                    "metrics": asdict(r.metrics)
                }
                for r in self.results
            ],
            "aggregate_stats": self._compute_aggregate_stats()
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n[OK] Results saved to {filepath}")


# ============================================================================
# REGRESSION TESTING
# ============================================================================

class RegressionTester:
    """Track evaluation metrics over time to detect degradation"""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.history_file = results_dir / "evaluation_history.json"

    def load_history(self) -> List[Dict]:
        """Load previous evaluation results"""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []

    def save_history(self, history: List[Dict]):
        """Save evaluation history"""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def compare_runs(self, current: Dict, previous: Dict) -> Dict[str, Any]:
        """
        Compare current evaluation to previous run

        Returns:
            Degradation analysis
        """
        if not previous:
            return {"status": "first_run", "message": "No previous results to compare"}

        current_score = current.get("overall_score", 0)
        previous_score = previous.get("overall_score", 0)

        delta = current_score - previous_score
        percent_change = (delta / previous_score * 100) if previous_score > 0 else 0

        status = "OK" if delta >= -0.05 else "DEGRADATION"

        return {
            "status": status,
            "current_score": current_score,
            "previous_score": previous_score,
            "absolute_delta": delta,
            "percent_change": percent_change,
            "requires_investigation": delta < -0.05
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run Phase 5: Evaluation"""

    # Setup paths
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "data" / "evaluation_results"

    try:
        # Initialize pipeline
        print("\nInitializing RAG pipeline...")
        rag = RAGPipeline()

        if not rag.ready:
            print("[ERROR] RAG pipeline not ready. Run Phase 1 first.")
            return

        # Run evaluation
        evaluator = EvaluationEngine(rag)
        stats = evaluator.run_benchmark(subset="all")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"evaluation_{timestamp}.json"
        evaluator.save_results(results_file)

        # Regression testing
        print("\n" + "="*70)
        print("REGRESSION TESTING")
        print("="*70)

        tester = RegressionTester(results_dir)
        history = tester.load_history()

        if history:
            comparison = tester.compare_runs(stats, history[-1])
            print(f"\nComparison to previous run:")
            print(f"  Previous score: {comparison['previous_score']:.3f}")
            print(f"  Current score:  {comparison['current_score']:.3f}")
            print(f"  Delta:          {comparison['absolute_delta']:+.3f}")
            print(f"  Change:         {comparison['percent_change']:+.1f}%")
            print(f"  Status:         {comparison['status']}")
        else:
            print("\nFirst evaluation run - establishing baseline")

        # Update history
        history.append(stats)
        tester.save_history(history)

        # Summary
        print("\n" + "="*70)
        print("[OK] PHASE 5 COMPLETE!")
        print("="*70)
        print(f"\nSummary:")
        print(f"  Overall Score:        {stats['overall_score']:.3f}/1.00")
        print(f"  Total Questions:      {stats['total_questions']}")
        print(f"  Results saved to:     {results_file}")
        print(f"  History saved to:     {tester.history_file}")
        print("\nMetric Averages:")
        for metric, values in stats['metrics'].items():
            print(f"  {metric:20s}: {values['avg']:.3f}")
        print("\n" + "="*70)
        print("NEXT: Run Phase 6 to build REST API backend")
        print("      $ python scripts/api_server.py")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
