"""Review templates stay blank; condition swaps cannot be rescued by real quotes."""
import json

import pytest

from src.eval import reader_comparison_review as reviewing
from tests.test_reader_comparison import frozen, scripted_run  # noqa: F401


@pytest.fixture
def reviewable(frozen, tmp_path, monkeypatch):  # noqa: F811
    plan, prepared, _, data = frozen
    monkeypatch.setattr(reviewing, 'DATA', data)
    run_dir, review_dir = tmp_path / 'run', tmp_path / 'review'
    results, _ = scripted_run(frozen, run_dir)
    reviewing.export(prepared, run_dir, review_dir)
    packet = json.loads((review_dir / 'review-packet.json').read_text())
    key = json.loads((review_dir / 'model-key.json').read_text())
    review = json.loads((review_dir / 'review-template.json').read_text())
    return plan, results, packet, key, review


def fill(reviewable):
    plan, results, packet, key, review = reviewable
    review.update(reviewer_kind='model_assisted', reviewer_identity='scripted test fixture',
                  reviewed_at='2026-10-20T00:00:00Z')
    for row in review['rows']:
        arm = key['key'][f"{row['case_id']}:{row['slot']}"]['arm']
        row.update(verdict='pass' if arm == 'v5' else 'fail', condition_error=False,
                   unsupported_numeric=False, notes='Scripted test grade; not an actual review.')
    for cell in results['cells']:
        cell['seconds'] = 1.0
    key['results_sha256'] = reviewing.digest(results)
    return plan, results, packet, key, review


def test_export_hides_models_and_does_not_fill_grades(reviewable):
    _, _, packet, _, review = reviewable
    raw = json.dumps(packet)
    assert 'adapter_applied' not in raw and 'envoy-base-only' not in raw
    assert 'model_identity' not in raw and 'served_model' not in raw
    assert all(r['verdict'] is None for r in review['rows'])
    with pytest.raises(ValueError, match='attributed'):
        reviewing.summarize(*reviewable)


def test_supported_and_gap_scores_separate_and_model_review_labelled(reviewable):
    report = reviewing.summarize(*fill(reviewable))
    assert report['arms']['v5']['supported_passes'] == 8
    assert report['arms']['v5']['gap_passes'] == 4
    assert report['decision'] == 'favorable_development_signal'
    assert report['human_review'] == 'not_performed'


@pytest.mark.parametrize('change', ['condition', 'missing', 'tokens', 'binding'])
def test_incomplete_or_invalid_judgments_cannot_claim_a_favorable_signal(reviewable, change):
    args = fill(reviewable)
    _, results, _, _, review = args
    if change == 'condition':
        next(r for r in review['rows'] if r['verdict'] == 'pass')['condition_error'] = True
    elif change == 'missing':
        review['rows'].pop()
    elif change == 'tokens':
        results['cells'][0]['audit']['reported_usage'] = {}
    else:
        results['cells'][0]['seconds'] += 1
    with pytest.raises(ValueError):
        reviewing.summarize(*args)


def test_positive_totals_with_missing_request_coverage_make_cost_inconclusive(reviewable):
    args = fill(reviewable)
    _, results, _, key, _ = args
    results['cells'][0]['audit']['usage_complete'] = False
    assert results['cells'][0]['audit']['reported_usage']['total_tokens'] > 0
    key['results_sha256'] = reviewing.digest(results)
    report = reviewing.summarize(*args)
    assert report['decision'] == 'inconclusive_missing_telemetry'


def test_same_review_cannot_score_a_different_plan(reviewable):
    args = fill(reviewable)
    args[0]['guidance'] = 'different'
    with pytest.raises(ValueError, match='binding'):
        reviewing.summarize(*args)


@pytest.mark.parametrize('change', ['runtime', 'schedule', 'references'])
def test_export_cannot_relabel_an_old_run_or_change_its_reference(frozen, tmp_path,  # noqa: F811
                                                               monkeypatch, change):
    plan, prepared, _, data = frozen
    monkeypatch.setattr(reviewing, 'DATA', data)
    run_dir = tmp_path / 'binding-run'
    results, _ = scripted_run(frozen, run_dir)
    if change == 'runtime':
        (run_dir / 'runtime.json').write_text('{"plan_sha256":"different"}')
    elif change == 'schedule':
        results['cells'].reverse()
        (run_dir / 'results.json').write_text(json.dumps(results))
    else:
        (prepared / 'references.json').write_text('{}')
    with pytest.raises((ValueError, KeyError)):
        reviewing.export(prepared, run_dir, tmp_path / 'rejected-review')
