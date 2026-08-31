"""Tests for entity linking."""

from finintel.nlp.entity_linker import EntityLinker


def test_ticker_extraction():
    linker = EntityLinker()
    entities = linker.extract_and_link(
        "AAPL stock surged 5% after Apple Inc reported strong earnings",
        query="AAPL",
    )
    tickers = linker.get_tickers(entities)
    assert "AAPL" in tickers


def test_company_alias():
    linker = EntityLinker()
    entities = linker.extract_and_link(
        "Microsoft announces new AI product line for enterprise customers",
        query="MSFT",
    )
    tickers = linker.get_tickers(entities)
    assert "MSFT" in tickers


def test_macro_entity():
    linker = EntityLinker()
    entities = linker.extract_and_link(
        "Federal Reserve signals potential interest rate cut amid inflation concerns",
    )
    names = [e.canonical_name for e in entities]
    assert "Federal Reserve" in names or "Inflation" in names


def test_regulator():
    linker = EntityLinker()
    entities = linker.extract_and_link(
        "SEC launches investigation into accounting practices",
    )
    names = [e.canonical_name for e in entities]
    assert "SEC" in names
