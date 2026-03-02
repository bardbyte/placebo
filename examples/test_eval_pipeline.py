#!/usr/bin/env python3
"""
Tests for the Evaluation Pipeline.

Tests schema discovery parsing, question generation parsing, metrics scoring,
and the full pipeline flow with mocks.

Usage:
    python examples/test_eval_pipeline.py -v
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from eval_pipeline import (
        LookerSchema,
        ExploreInfo,
        FieldInfo,
        GeneratedQuestion,
        QueryDifficulty,
        EvalMetrics,
        DetailedResult,
        discover_schema,
    )
    from agent_ca import CAResponse, parse_ca_response
    FULL_IMPORT = True
except ImportError as e:
    print(f"  Import error: {e}")
    print("  Some tests will be skipped")
    FULL_IMPORT = False


# ===========================================================================
# Test infrastructure (reuse from test_agent_ca.py)
# ===========================================================================

class TestResult:
    def __init__(self, name, passed, message="", duration_ms=0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration_ms = duration_ms


class TestSuite:
    def __init__(self, name, verbose=False):
        self.name = name
        self.results = []
        self.verbose = verbose

    def run_test(self, name, test_fn):
        start = time.perf_counter()
        try:
            test_fn()
            duration = (time.perf_counter() - start) * 1000
            self.results.append(TestResult(name, True, duration_ms=duration))
            if self.verbose:
                print(f"  \033[92mPASS\033[0m {name} ({duration:.0f}ms)")
        except AssertionError as e:
            duration = (time.perf_counter() - start) * 1000
            self.results.append(TestResult(name, False, str(e), duration))
            print(f"  \033[91mFAIL\033[0m {name}: {e}")
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            self.results.append(TestResult(name, False, f"Exception: {e}", duration))
            print(f"  \033[91mERROR\033[0m {name}: {e}")

    async def run_test_async(self, name, test_fn):
        start = time.perf_counter()
        try:
            await test_fn()
            duration = (time.perf_counter() - start) * 1000
            self.results.append(TestResult(name, True, duration_ms=duration))
            if self.verbose:
                print(f"  \033[92mPASS\033[0m {name} ({duration:.0f}ms)")
        except AssertionError as e:
            duration = (time.perf_counter() - start) * 1000
            self.results.append(TestResult(name, False, str(e), duration))
            print(f"  \033[91mFAIL\033[0m {name}: {e}")
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            self.results.append(TestResult(name, False, f"Exception: {e}", duration))
            print(f"  \033[91mERROR\033[0m {name}: {e}")

    def summary(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        print(f"\n  {self.name}: {passed}/{total} passed", end="")
        if passed == total:
            print(" \033[92m✓\033[0m")
        else:
            print(f" \033[91m({total - passed} failed)\033[0m")
        return passed == total


def assert_eq(actual, expected, msg=""):
    if actual != expected:
        raise AssertionError(f"{msg}: expected {expected!r}, got {actual!r}")


def assert_true(val, msg=""):
    if not val:
        raise AssertionError(f"{msg}: expected truthy, got {val!r}")


def assert_false(val, msg=""):
    if val:
        raise AssertionError(f"{msg}: expected falsy, got {val!r}")


def assert_gt(actual, threshold, msg=""):
    if not actual > threshold:
        raise AssertionError(f"{msg}: expected > {threshold}, got {actual}")


def assert_near(actual, expected, tolerance=0.1, msg=""):
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{msg}: expected ~{expected}, got {actual} (tolerance {tolerance})")


# ===========================================================================
# Schema Tests
# ===========================================================================

def build_test_schema() -> LookerSchema:
    """Build a realistic test schema."""
    return LookerSchema(
        discovered_at="2026-03-01T00:00:00Z",
        explores=[
            ExploreInfo(
                model="finance",
                name="transactions",
                label="Transactions",
                description="All card transactions",
                dimensions=[
                    FieldInfo("transactions.id", "ID", "dimension", "number", "Transaction ID"),
                    FieldInfo("transactions.date", "Date", "dimension", "date", "Transaction date"),
                    FieldInfo("transactions.region", "Region", "dimension", "string", "Geographic region"),
                    FieldInfo("transactions.category", "Category", "dimension", "string", "Merchant category"),
                    FieldInfo("transactions.card_type", "Card Type", "dimension", "string", "Type of card used"),
                ],
                measures=[
                    FieldInfo("transactions.total_amount", "Total Amount", "measure", "sum", "Sum of transaction amounts"),
                    FieldInfo("transactions.count", "Count", "measure", "count", "Number of transactions"),
                    FieldInfo("transactions.avg_amount", "Average Amount", "measure", "average", "Average transaction amount"),
                ],
            ),
            ExploreInfo(
                model="finance",
                name="accounts",
                label="Accounts",
                description="Customer accounts",
                dimensions=[
                    FieldInfo("accounts.id", "ID", "dimension", "number", "Account ID"),
                    FieldInfo("accounts.status", "Status", "dimension", "string", "Active/Closed"),
                    FieldInfo("accounts.opened_date", "Opened Date", "dimension", "date", "Account open date"),
                    FieldInfo("accounts.state", "State", "dimension", "string", "Customer state"),
                ],
                measures=[
                    FieldInfo("accounts.count", "Count", "measure", "count", "Number of accounts"),
                    FieldInfo("accounts.total_balance", "Total Balance", "measure", "sum", "Sum of balances"),
                ],
            ),
        ],
    )


def run_schema_tests(verbose=False):
    suite = TestSuite("Schema Model", verbose)
    schema = build_test_schema()

    def test_models():
        assert_eq(schema.models, ["finance"], "models")

    def test_explore_count():
        assert_eq(len(schema.explores), 2, "explore count")

    def test_total_dimensions():
        assert_eq(schema.total_dimensions, 9, "total dims")

    def test_total_measures():
        assert_eq(schema.total_measures, 5, "total measures")

    def test_fingerprint():
        fp = schema.fingerprint()
        assert_eq(len(fp), 12, "fingerprint length")
        # Same schema = same fingerprint
        fp2 = schema.fingerprint()
        assert_eq(fp, fp2, "fingerprint deterministic")

    def test_fingerprint_changes():
        schema2 = build_test_schema()
        schema2.explores[0].dimensions.append(
            FieldInfo("transactions.new_field", "New", "dimension", "string", "")
        )
        assert_true(schema.fingerprint() != schema2.fingerprint(), "fingerprint changes on schema change")

    def test_explore_date_dimensions():
        exp = schema.explores[0]
        dates = exp.date_dimensions
        assert_eq(len(dates), 1, "date dims count")
        assert_eq(dates[0].name, "transactions.date", "date dim name")

    def test_explore_string_dimensions():
        exp = schema.explores[0]
        strings = exp.string_dimensions
        assert_eq(len(strings), 3, "string dims count")

    def test_explore_numeric_measures():
        exp = schema.explores[0]
        nums = exp.numeric_measures
        assert_eq(len(nums), 3, "numeric measures count")

    def test_schema_summary():
        exp = schema.explores[0]
        summary = exp.schema_summary()
        assert_true("transactions" in summary, "has explore name")
        assert_true("transactions.total_amount" in summary, "has measure name")
        assert_true("transactions.region" in summary, "has dim name")

    def test_to_dict():
        d = schema.to_dict()
        assert_eq(len(d["explores"]), 2, "explores in dict")
        assert_eq(d["explores"][0]["dimension_count"], 5, "dim count in dict")
        assert_eq(d["explores"][0]["measure_count"], 3, "measure count in dict")

    suite.run_test("models property", test_models)
    suite.run_test("explore count", test_explore_count)
    suite.run_test("total dimensions", test_total_dimensions)
    suite.run_test("total measures", test_total_measures)
    suite.run_test("fingerprint", test_fingerprint)
    suite.run_test("fingerprint changes", test_fingerprint_changes)
    suite.run_test("date dimensions", test_explore_date_dimensions)
    suite.run_test("string dimensions", test_explore_string_dimensions)
    suite.run_test("numeric measures", test_explore_numeric_measures)
    suite.run_test("schema summary", test_schema_summary)
    suite.run_test("to_dict", test_to_dict)

    return suite.summary()


# ===========================================================================
# Question Model Tests
# ===========================================================================

def run_question_tests(verbose=False):
    suite = TestSuite("Question Model", verbose)

    def test_roundtrip():
        q = GeneratedQuestion(
            id="simple_001",
            query="What is the total transaction amount by region?",
            difficulty=QueryDifficulty.SIMPLE,
            category="basic_aggregation",
            target_model="finance",
            target_explore="transactions",
            expected_dimensions=["transactions.region"],
            expected_measures=["transactions.total_amount"],
            expected_filters={},
            notes="Basic group-by",
        )
        d = q.to_dict()
        q2 = GeneratedQuestion.from_dict(d)
        assert_eq(q2.id, q.id, "id roundtrip")
        assert_eq(q2.query, q.query, "query roundtrip")
        assert_eq(q2.difficulty, q.difficulty, "difficulty roundtrip")
        assert_eq(q2.target_model, q.target_model, "model roundtrip")
        assert_eq(q2.expected_dimensions, q.expected_dimensions, "dims roundtrip")

    def test_all_difficulties():
        for diff in QueryDifficulty:
            q = GeneratedQuestion(
                id=f"{diff.value}_001",
                query="test",
                difficulty=diff,
                category="test",
                target_model="m",
                target_explore="e",
            )
            d = q.to_dict()
            assert_eq(d["difficulty"], diff.value, f"{diff.value} serializes")
            q2 = GeneratedQuestion.from_dict(d)
            assert_eq(q2.difficulty, diff, f"{diff.value} deserializes")

    def test_empty_expectations():
        q = GeneratedQuestion(
            id="adv_001",
            query="What's the weather?",
            difficulty=QueryDifficulty.ADVERSARIAL,
            category="out_of_scope",
            target_model="",
            target_explore="",
        )
        d = q.to_dict()
        assert_eq(d["expected_dimensions"], [], "empty dims")
        assert_eq(d["expected_measures"], [], "empty measures")

    suite.run_test("roundtrip serialization", test_roundtrip)
    suite.run_test("all difficulties", test_all_difficulties)
    suite.run_test("empty expectations", test_empty_expectations)

    return suite.summary()


# ===========================================================================
# Metrics Scoring Tests
# ===========================================================================

def run_metrics_tests(verbose=False):
    suite = TestSuite("Eval Metrics", verbose)

    def make_ca_response(answer=None, model=None, explore=None, fields=None,
                          filters=None, data=None, error=None):
        return CAResponse(
            answer=answer,
            retrieval_query={
                "model": model,
                "view": explore,
                "fields": fields or [],
                "filters": filters or {},
            } if model else None,
            data_retrieved=data,
            error=error,
        )

    def test_perfect_score():
        metrics = EvalMetrics()
        q = GeneratedQuestion(
            id="s1", query="Revenue by region", difficulty=QueryDifficulty.SIMPLE,
            category="agg", target_model="finance", target_explore="transactions",
            expected_dimensions=["transactions.region"],
            expected_measures=["transactions.total_amount"],
        )
        ca = make_ca_response(
            answer="Here's revenue by region",
            model="finance", explore="transactions",
            fields=["transactions.region", "transactions.total_amount"],
            data=[{"region": "West", "amount": 100}],
        )
        metrics.record(q, ca, 1500.0)

        assert_eq(metrics.total_queries, 1, "total")
        assert_eq(metrics.successful_queries, 1, "successful")
        assert_near(metrics.success_rate, 100.0, msg="success rate")
        assert_near(metrics.model_accuracy, 100.0, msg="model acc")
        assert_near(metrics.explore_accuracy, 100.0, msg="explore acc")
        assert_near(metrics.field_accuracy, 100.0, msg="field acc")

    def test_wrong_model():
        metrics = EvalMetrics()
        q = GeneratedQuestion(
            id="s1", query="test", difficulty=QueryDifficulty.SIMPLE,
            category="agg", target_model="finance", target_explore="transactions",
            expected_dimensions=["transactions.region"],
            expected_measures=["transactions.total_amount"],
        )
        ca = make_ca_response(
            answer="answer", model="marketing", explore="campaigns",
            fields=["campaigns.region", "campaigns.spend"],
            data=[{"region": "West"}],
        )
        metrics.record(q, ca, 1000.0)
        assert_near(metrics.model_accuracy, 0.0, msg="wrong model = 0%")
        assert_near(metrics.explore_accuracy, 0.0, msg="wrong explore = 0%")

    def test_error_query():
        metrics = EvalMetrics()
        q = GeneratedQuestion(
            id="s1", query="test", difficulty=QueryDifficulty.SIMPLE,
            category="agg", target_model="finance", target_explore="transactions",
        )
        ca = make_ca_response(error="Something broke")
        metrics.record(q, ca, 500.0)
        assert_eq(metrics.error_queries, 1, "errors")
        assert_near(metrics.success_rate, 0.0, msg="success rate with error")

    def test_adversarial_skips_accuracy():
        metrics = EvalMetrics()
        q = GeneratedQuestion(
            id="a1", query="What's the weather?", difficulty=QueryDifficulty.ADVERSARIAL,
            category="out_of_scope", target_model="", target_explore="",
        )
        ca = make_ca_response(answer="I can only help with data questions")
        metrics.record(q, ca, 300.0)
        assert_eq(metrics.has_expected_count, 0, "adversarial not counted for accuracy")

    def test_latency_percentiles():
        metrics = EvalMetrics()
        q = GeneratedQuestion(
            id="s1", query="test", difficulty=QueryDifficulty.SIMPLE,
            category="agg", target_model="m", target_explore="e",
        )
        latencies = [100, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 5000]
        for lat in latencies:
            ca = make_ca_response(answer="ok", model="m", explore="e", data=[{"a": 1}])
            metrics.record(q, ca, lat)

        assert_eq(metrics.total_queries, 10, "total")
        assert_true(metrics.latency_p50 <= 1000, f"p50={metrics.latency_p50}")
        assert_true(metrics.latency_p90 >= 2000, f"p90={metrics.latency_p90}")
        assert_true(metrics.latency_p99 >= 3000, f"p99={metrics.latency_p99}")

    def test_by_difficulty_breakdown():
        metrics = EvalMetrics()
        for diff in [QueryDifficulty.SIMPLE, QueryDifficulty.SIMPLE, QueryDifficulty.MODERATE]:
            q = GeneratedQuestion(
                id="x", query="test", difficulty=diff,
                category="agg", target_model="m", target_explore="e",
                expected_dimensions=["e.dim"],
            )
            ca = make_ca_response(
                answer="ok", model="m", explore="e",
                fields=["e.dim"], data=[{"dim": "a"}],
            )
            metrics.record(q, ca, 1000.0)

        assert_eq(metrics.by_difficulty["simple"]["total"], 2, "simple count")
        assert_eq(metrics.by_difficulty["moderate"]["total"], 1, "moderate count")

    def test_by_category_breakdown():
        metrics = EvalMetrics()
        for cat in ["aggregation", "aggregation", "time_filter"]:
            q = GeneratedQuestion(
                id="x", query="test", difficulty=QueryDifficulty.SIMPLE,
                category=cat, target_model="m", target_explore="e",
            )
            ca = make_ca_response(answer="ok", model="m", explore="e", data=[{"a": 1}])
            metrics.record(q, ca, 1000.0)

        assert_eq(metrics.by_category["aggregation"]["total"], 2, "agg category")
        assert_eq(metrics.by_category["time_filter"]["total"], 1, "time category")

    def test_partial_fields():
        metrics = EvalMetrics()
        q = GeneratedQuestion(
            id="s1", query="test", difficulty=QueryDifficulty.SIMPLE,
            category="agg", target_model="m", target_explore="e",
            expected_dimensions=["e.dim1", "e.dim2"],
            expected_measures=["e.measure1"],
        )
        ca = make_ca_response(
            answer="ok", model="m", explore="e",
            fields=["e.dim1", "e.measure1"],  # missing e.dim2
            data=[{"a": 1}],
        )
        metrics.record(q, ca, 1000.0)
        assert_eq(metrics.partial_fields_count, 1, "partial fields detected")
        assert_eq(metrics.correct_fields_count, 0, "not fully correct")

    def test_empty_results_counted_as_success():
        metrics = EvalMetrics()
        q = GeneratedQuestion(
            id="s1", query="test", difficulty=QueryDifficulty.SIMPLE,
            category="agg", target_model="m", target_explore="e",
        )
        ca = make_ca_response(
            answer="No data found for that filter",
            model="m", explore="e", fields=["e.dim"],
        )
        # data_retrieved is None but has query — system worked, just no matching data
        metrics.record(q, ca, 1000.0)
        assert_eq(metrics.successful_queries, 1, "empty result = success")

    def test_to_dict_structure():
        metrics = EvalMetrics()
        q = GeneratedQuestion(
            id="s1", query="test", difficulty=QueryDifficulty.SIMPLE,
            category="agg", target_model="m", target_explore="e",
            expected_dimensions=["e.dim"],
        )
        ca = make_ca_response(answer="ok", model="m", explore="e",
                               fields=["e.dim"], data=[{"a": 1}])
        metrics.record(q, ca, 1500.0)
        d = metrics.to_dict()

        assert_true("summary" in d, "has summary")
        assert_true("accuracy" in d, "has accuracy")
        assert_true("latency" in d, "has latency")
        assert_true("by_difficulty" in d, "has by_difficulty")
        assert_true("by_category" in d, "has by_category")
        assert_eq(d["summary"]["total_queries"], 1, "total in dict")

    suite.run_test("perfect score", test_perfect_score)
    suite.run_test("wrong model", test_wrong_model)
    suite.run_test("error query", test_error_query)
    suite.run_test("adversarial skips accuracy", test_adversarial_skips_accuracy)
    suite.run_test("latency percentiles", test_latency_percentiles)
    suite.run_test("by difficulty breakdown", test_by_difficulty_breakdown)
    suite.run_test("by category breakdown", test_by_category_breakdown)
    suite.run_test("partial fields", test_partial_fields)
    suite.run_test("empty results = success", test_empty_results_counted_as_success)
    suite.run_test("to_dict structure", test_to_dict_structure)

    return suite.summary()


# ===========================================================================
# Mock Schema Discovery Tests
# ===========================================================================

async def run_discovery_tests(verbose=False):
    suite = TestSuite("Schema Discovery (Mock)", verbose)

    if not FULL_IMPORT:
        print("  SKIPPED (requires full import)")
        return True

    from lumi_llm.mcp.client import MCPTool, MCPToolResult

    class MockMCPClient:
        def __init__(self):
            self.tools = [
                MCPTool(name="get-models", description="", input_schema={}),
                MCPTool(name="get-explores", description="", input_schema={}),
                MCPTool(name="get-dimensions", description="", input_schema={}),
                MCPTool(name="get-measures", description="", input_schema={}),
            ]

        async def call_tool(self, name, arguments):
            if name == "get-models":
                return MCPToolResult(content=json.dumps([
                    {"name": "finance", "label": "Finance"},
                ]))
            elif name == "get-explores":
                return MCPToolResult(content=json.dumps([
                    {"name": "transactions", "label": "Transactions",
                     "description": "Card transactions"},
                ]))
            elif name == "get-dimensions":
                return MCPToolResult(content=json.dumps([
                    {"name": "transactions.date", "label": "Date",
                     "type": "date", "description": "Transaction date", "tags": []},
                    {"name": "transactions.region", "label": "Region",
                     "type": "string", "description": "Geographic region", "tags": ["geo"]},
                ]))
            elif name == "get-measures":
                return MCPToolResult(content=json.dumps([
                    {"name": "transactions.total_amount", "label": "Total Amount",
                     "type": "sum", "description": "Sum of amounts", "tags": ["kpi"]},
                    {"name": "transactions.count", "label": "Count",
                     "type": "count", "description": "Transaction count", "tags": []},
                ]))
            return MCPToolResult(content="", is_error=True)

    mock = MockMCPClient()

    async def test_discovery():
        schema = await discover_schema(mock, verbose=False)
        assert_eq(len(schema.explores), 1, "one explore")
        assert_eq(schema.explores[0].model, "finance", "model name")
        assert_eq(schema.explores[0].name, "transactions", "explore name")
        assert_eq(len(schema.explores[0].dimensions), 2, "2 dims")
        assert_eq(len(schema.explores[0].measures), 2, "2 measures")
        assert_eq(schema.total_dimensions, 2, "total dims")
        assert_eq(schema.total_measures, 2, "total measures")

    async def test_field_types():
        schema = await discover_schema(mock, verbose=False)
        exp = schema.explores[0]
        dates = exp.date_dimensions
        assert_eq(len(dates), 1, "1 date dim")
        strings = exp.string_dimensions
        assert_eq(len(strings), 1, "1 string dim")

    async def test_fingerprint_stable():
        s1 = await discover_schema(mock, verbose=False)
        s2 = await discover_schema(mock, verbose=False)
        assert_eq(s1.fingerprint(), s2.fingerprint(), "same schema = same fingerprint")

    await suite.run_test_async("full discovery", test_discovery)
    await suite.run_test_async("field types", test_field_types)
    await suite.run_test_async("fingerprint stable", test_fingerprint_stable)

    return suite.summary()


# ===========================================================================
# Main
# ===========================================================================

async def main():
    verbose = "-v" in sys.argv or "--verbose" in sys.argv

    print("=" * 60)
    print("  Evaluation Pipeline — Test Suite")
    print("=" * 60)

    all_passed = True

    if FULL_IMPORT:
        print("\n--- Schema Model Tests ---")
        all_passed &= run_schema_tests(verbose)

        print("\n--- Question Model Tests ---")
        all_passed &= run_question_tests(verbose)

        print("\n--- Eval Metrics Tests ---")
        all_passed &= run_metrics_tests(verbose)

        print("\n--- Schema Discovery (Mock) ---")
        all_passed &= await run_discovery_tests(verbose)
    else:
        print("\n  Cannot run tests — import failed")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("  \033[92mALL TESTS PASSED\033[0m")
    else:
        print("  \033[91mSOME TESTS FAILED\033[0m")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
