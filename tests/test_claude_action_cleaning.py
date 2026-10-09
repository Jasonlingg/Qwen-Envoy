import ast

from src.policies.claude_action_cleaning import clean_action


def test_uniformly_indented_code_is_dedented_not_stripped_on_one_line_only():
    """Regression: a global .strip() before dedent only removes line 1's
    indentation, leaving later lines indented -> IndentationError. This was a
    real cause of Sonnet 5 failures in the QASPER comparison eval."""
    raw = '    r = search_within("d1", "x")\n    print(r)'
    cleaned = clean_action(raw)
    ast.parse(cleaned)  # must not raise
    assert cleaned == 'r = search_within("d1", "x")\nprint(r)'


def test_xml_tool_call_wrapper_is_stripped():
    raw = (
        "I'll look this up.\n"
        '<function_calls>\n<invoke name="search">\n'
        '<parameter name="query">nodes</parameter>\n</invoke>\n</function_calls>'
    )
    cleaned = clean_action(raw)
    assert "<function_calls>" not in cleaned
    assert "I'll look this up" not in cleaned


def test_hallucinated_multiturn_dump_with_no_real_code_returns_empty_not_prose():
    """Regression: when nothing survives the code-line filter (e.g. tag-stripped
    XML parameter fragments with no assignment/call shape), the old fallback
    returned the entire unfiltered text, which then failed as a multi-line
    SyntaxError instead of a clean empty action."""
    raw = (
        "Let me search for this.\n"
        '<function_calls>\n<invoke name="read">\n'
        '<parameter name="doc_id">qasper_123</parameter>\n</invoke>\n</function_calls>\n'
        "Based on that, let me check further.\n"
        '<function_calls>\n<invoke name="search_within">\n'
        '<parameter name="doc_id">qasper_123</parameter>\n'
        '<parameter name="query">hyperparameters</parameter>\n</invoke>\n</function_calls>'
    )
    cleaned = clean_action(raw)
    assert cleaned.strip() == ""


def test_real_code_survives_alongside_dropped_prose():
    raw = "Let me check the paper.\nr = search(\"nodes\")\nprint(r)\nThat should tell us."
    cleaned = clean_action(raw)
    assert cleaned == 'r = search("nodes")\nprint(r)'
    ast.parse(cleaned)


def test_submit_line_is_extracted_even_from_a_prose_and_xml_wrapper():
    raw = "Based on my research, here is the answer.\nSUBMIT: No CITATIONS: [\"d1\"]"
    assert clean_action(raw) == 'SUBMIT: No CITATIONS: ["d1"]'


def test_submit_keeps_exact_evidence_after_citations():
    raw = (
        'SUBMIT: supported fact CITATIONS: ["d1"] '
        'EVIDENCE: [{"doc_id":"d1","start":3,"end":17}]'
    )
    assert clean_action(raw) == raw


def test_multistep_dump_is_truncated_to_the_first_step():
    raw = "# Step 1\nr = search(\"a\")\nprint(r)\n# Step 2\nread(\"b\")"
    cleaned = clean_action(raw)
    assert "Step 2" not in cleaned
    assert 'search("a")' in cleaned
