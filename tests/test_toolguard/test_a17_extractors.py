"""A1.7 unit tests: deterministic source/state/topic/ID extractors and their opt-in.

The extractors are conservative by design: when the task text does not contain an
explicit value they return no candidate instead of guessing.  The failure cases below
are taken from the real 228-case A1.7 subset.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tool_evolution.toolguard.engine import repair_failure  # noqa: E402
from tool_evolution.toolguard.extractors import (  # noqa: E402
    EXTRACTORS, EXPERIMENTAL_EXTRACTORS, ExtractorMap, extract_identifiers, extract_sources,
    extract_states, extract_topics,
)
from tool_evolution.toolguard.records import FunctionSchema, ToolCall  # noqa: E402

# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
COINDESK_QUERY = (
    "I'm interested in the latest news about cryptocurrencies. Can you fetch the top "
    "articles from CoinDesk, BSC News, and Decrypt? Please include the article titles "
    "and publication dates."
)


def test_extract_sources_reads_a_from_introduced_publisher_list():
    assert extract_sources(COINDESK_QUERY) == ["CoinDesk", "BSC News", "Decrypt"]


def test_extract_sources_keeps_the_guardian_style_multiword_names():
    text = ("Can you fetch the top news articles from CoinDesk, The Guardian, and Decrypt?")
    assert extract_sources(text) == ["CoinDesk", "The Guardian", "Decrypt"]


def test_extract_sources_failure_case_has_no_capitalised_source():
    # failure sample from the 228 subset: "from a specific URL" names no publisher
    text = ("I'm planning a family vacation. Can you fetch the content from a specific URL: "
            "'https://www.example.com'?")
    assert extract_sources(text) == []


def test_extract_sources_does_not_treat_lowercase_prose_as_a_source():
    assert extract_sources("Please search from the latest news about dogs.") == []
    assert extract_sources("") == []


# ---------------------------------------------------------------------------
# states
# ---------------------------------------------------------------------------


def test_extract_states_returns_code_and_full_name():
    assert extract_states("What is the weather in California next week?") == ["CA", "California"]
    # the curated vocabulary is scanned in declaration order (same convention as countries)
    assert extract_states("Compare Texas and New York") == ["NY", "New York", "TX", "Texas"]


def test_extract_states_failure_case_names_only_a_country():
    # failure sample: the COVID-19 India tool needs a state, the task text has none
    text = ("I'm interested in the COVID-19 situation in India. Can you provide the latest "
            "updates on active, recovered, and deaths cases in the country?")
    assert extract_states(text) == []


# ---------------------------------------------------------------------------
# topics
# ---------------------------------------------------------------------------


def test_extract_topics_reads_the_content_noun_modifier():
    text = ("My friends and I are starting a cryptocurrency blog and we need content. Can "
            "you fetch the top news articles from CoinDesk, The Guardian, and Decrypt?")
    assert extract_topics(text) == ["cryptocurrency"]


def test_extract_topics_failure_case_is_a_bare_content_phrase():
    # no topic is stated; "top news articles" must not invent one
    assert extract_topics("Can you fetch the top news articles for me?") == []
    assert extract_topics("") == []


# ---------------------------------------------------------------------------
# identifiers
# ---------------------------------------------------------------------------


def test_extract_identifiers_reads_imdb_ids_and_labelled_ids():
    assert extract_identifiers(
        "The IMDb ID of the series is tt1234567 and the season number is 3."
    ) == ["tt1234567"]
    assert extract_identifiers("please use track_id=42 here") == ["42"]
    assert extract_identifiers(
        "episode id: e-0001 and track 550e8400-e29b-41d4-a716-446655440000"
    ) == ["550e8400-e29b-41d4-a716-446655440000", "e-0001"]


def test_extract_identifiers_failure_case_has_no_bare_id_value():
    # failure sample: the tool needs an ID but the task text only says "a specific music ID"
    text = ("I'm curious about the popularity of a specific hashtag on TikTok. Can you fetch "
            "me the music data for a specific music ID.")
    assert extract_identifiers(text) == []


# ---------------------------------------------------------------------------
# opt-in plumbing: defaults unchanged, experimental map enabled explicitly
# ---------------------------------------------------------------------------


def _success(tool: str, arguments: dict, query: str) -> ToolCall:
    return ToolCall(
        trace_id="m/G3/try0", method="m", model="chatgpt", strategy="cot",
        group="G3_instruction", query_id="1", try_index=0, step=1,
        tool_name=tool, arguments=arguments, error=None, recorded_ok=True,
        available_tools=[tool], valid_data=True, query=query,
    )


def test_default_extractor_set_and_map_do_not_include_a17_extractors():
    assert set(EXPERIMENTAL_EXTRACTORS) == {"sources", "states", "topics", "identifiers"}
    assert not set(EXPERIMENTAL_EXTRACTORS) & set(EXTRACTORS)
    mapping = ExtractorMap()
    for name in ("source", "state", "topic", "ID", "music_id"):
        assert mapping.extractor_for(name) is None, name
    # default learning over source-valued successes must not learn an experimental extractor
    successes = [
        _success("coindesk_for_cryptocurrency_news", {"source": "CoinDesk"}, COINDESK_QUERY)
        for _ in range(4)
    ]
    learned = ExtractorMap.learn(successes, {})
    assert learned.by_param.get("source") is None
    assert not any(kind in EXPERIMENTAL_EXTRACTORS for kind in learned.by_param.values())


def test_experimental_map_enables_the_name_fallbacks_and_candidates():
    mapping = ExtractorMap(include_experimental=True)
    assert mapping.extractor_for("source") == "sources"
    assert mapping.extractor_for("state") == "states"
    assert mapping.extractor_for("topic") == "topics"
    assert mapping.extractor_for("ID") == "identifiers"
    assert mapping.candidates("source", COINDESK_QUERY) == ["CoinDesk", "BSC News", "Decrypt"]
    # deterministic extractor names are registered in the pool
    assert set(EXPERIMENTAL_EXTRACTORS) <= set(ExtractorMap(include_experimental=True)._pool)


def test_experimental_learning_can_learn_source_extractor_from_training_successes():
    successes = [
        _success("coindesk_for_cryptocurrency_news", {"source": "CoinDesk"}, COINDESK_QUERY)
        for _ in range(4)
    ]
    learned = ExtractorMap.learn(successes, {}, include_experimental=True)
    assert learned.extractor_for("source") == "sources"


# ---------------------------------------------------------------------------
# engine integration: opt-in only
# ---------------------------------------------------------------------------

SOURCE_SCHEMA = FunctionSchema(
    name="coindesk_for_cryptocurrency_news",
    properties={"source": {"type": "string"}},
    required=("source",),
)


def _failure(tool: str, arguments: dict, error: str, query: str) -> ToolCall:
    return ToolCall(
        trace_id="m/G3/try0", method="m", model="gpt-4-0613", strategy="cot",
        group="G3_instruction", query_id="1", try_index=0, step=1,
        tool_name=tool, arguments=arguments, error=error, recorded_ok=False,
        available_tools=[tool], valid_data=True, query=query,
    )


def test_default_map_does_not_fill_source_but_experimental_map_does():
    call = _failure("coindesk_for_cryptocurrency_news", {},
                    "Missing required parameters: 'source'", COINDESK_QUERY)
    default = repair_failure(call, {"coindesk_for_cryptocurrency_news": SOURCE_SCHEMA},
                             None, ExtractorMap())
    assert not default.proposed
    enabled = repair_failure(call, {"coindesk_for_cryptocurrency_news": SOURCE_SCHEMA},
                             None, ExtractorMap(include_experimental=True))
    assert enabled.repairable and enabled.schema_valid
    assert enabled.arguments == {"source": "CoinDesk"}
    assert "text_extract:source" in enabled.rules


GUARDIAN_SCHEMA = FunctionSchema(
    name="the_guardian_for_cryptocurrency_news",
    properties={"topic": {}, "dateRange": {}},
    required=("topic", "dateRange"),
)


def test_missing_date_range_is_not_fabricated_when_the_text_has_no_date():
    call = _failure(
        "the_guardian_for_cryptocurrency_news", {},
        "Missing input parameters: 'topic', 'dateRange'.",
        "My friends and I are starting a cryptocurrency blog and we need content. Can you "
        "fetch the top news articles from CoinDesk, The Guardian, and Decrypt?")
    result = repair_failure(call, {"the_guardian_for_cryptocurrency_news": GUARDIAN_SCHEMA},
                            None, ExtractorMap(include_experimental=True))
    assert result.proposed, "topic extraction should propose a repair"
    assert not result.repairable, "the proposal must not pass while dateRange has no value"
    rejected = " ".join(result.rejected_proposals)
    assert "dateRange" in rejected
