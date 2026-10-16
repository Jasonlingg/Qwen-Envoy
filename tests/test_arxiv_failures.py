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


@pytest.mark.parametrize('tool,args,message', [
    ('open_paper', {'arxiv_id': '10.1145/123'}, 'modern arXiv ID'),
    ('get_passage', {'arxiv_id': '2609.12345v2', 'length': 9000}, '1..8000'),
    ('find_evidence', {'query': 'q', 'arxiv_ids': ['2609.12345v2']}, 'not open'),
    ('search_papers', {'query': '', 'limit': 0}, 'query must contain'),
    ('get_citations', {'arxiv_id': '2609.12345', 'direction': 'wrong'}, 'direction'),
])
def test_validation_messages_reach_mcp_client(tmp_path, tool, args, message):
    from src.product.paper_discovery import PaperDiscovery
    library, _ = make_library(tmp_path, {})
    server = create_server(library, PaperDiscovery(tmp_path / 'discovery'))
    async def body(client):
        return await client.call_tool(tool, args)
    result = _run(server, body)
    assert result.is_error
    assert message in ' '.join(getattr(item, 'text', '') for item in result.content)


def test_invalid_arxiv_metadata_is_source_status(tmp_path):
    from src.product.arxiv_library import ArxivLibrary
    library = ArxivLibrary(tmp_path / 'library',
        fetch=lambda _: b'<html>arXiv is temporarily unavailable</html>')
    server = create_server(library, FakeDiscovery())
    async def body(client):
        return await client.call_tool('open_paper', {'arxiv_id': '2609.12345'})
    result = _run(server, body)
    assert not result.is_error
    assert result.structured_content['status'] == 'source_unavailable'
    assert result.structured_content['source'] == 'arxiv'
