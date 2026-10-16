"""MCP failure paths preserve already verified evidence and bound input first."""
import pytest

from scripts.arxiv_research_mcp import create_server
from tests.arxiv_fakes import FakeInvestigator, article_page, make_library
from tests.test_arxiv_research_mcp import QUOTE, FakeDiscovery, _run


def _server(tmp_path, reasoner=None):
    library, fake = make_library(tmp_path, {'2609.12345v2': {'html': article_page()}})
    return create_server(library, FakeDiscovery(), investigator=FakeInvestigator(
        {'2609.12345v2': QUOTE}), reasoner=reasoner), fake


def test_failed_ingest_does_not_discard_other_evidence(tmp_path):
    server, _ = _server(tmp_path)
    async def body(client):
        return (await client.call_tool('investigate', {'question': 'q',
            'arxiv_ids': ['2609.99999v1', '2609.12345v2']})).structured_content
    result = _run(server, body)
    assert result['status'] == 'evidence_found' and len(result['evidence']) == 1
    assert result['papers'][0]['status'] == 'source_unavailable'
    assert result['paper_versions'] == ['2609.12345v2']


@pytest.mark.parametrize('failure', [RuntimeError, TimeoutError, KeyError])
def test_reasoner_exceptions_do_not_discard_verified_evidence(tmp_path, failure):
    class Reasoner:
        def synthesize(self, *_):
            raise failure('provider failed')
    server, _ = _server(tmp_path, Reasoner())
    async def body(client):
        return (await client.call_tool('investigate', {'question': 'q',
            'arxiv_ids': ['2609.12345v2'], 'synthesize': True})).structured_content
    result = _run(server, body)
    assert len(result['evidence']) == 1 and result['synthesis'] is None
    assert failure.__name__ in result['synthesis_error']


@pytest.mark.parametrize('question,ids', [('', ['2609.12345v2']),
    ('q', ['https://arxiv.org/abs/2609.12345']), ('q', ['2600.12345'])])
def test_bad_request_is_rejected_before_fetch(tmp_path, question, ids):
    server, fake = _server(tmp_path)
    async def body(client):
        return await client.call_tool('investigate', {'question': question, 'arxiv_ids': ids})
    assert _run(server, body).is_error
    assert not fake.calls
