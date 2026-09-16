import copy
import importlib
import json
from pathlib import Path

import jsonschema
import pytest


def validate(result, **context):
    import clothing_assistant.agent.pro as pro
    assert importlib.util.find_spec(pro.__name__ + '.validation'), 'Task7 validator missing'
    module = importlib.import_module(pro.__name__ + '.validation')
    return module.validate_pro_result(result, context={'run_id': 'r', 'request_id': 'q', 'thread_id': 't', **context})


def fixture():
    return {
        'run_id': 'r', 'query': '推荐棉质衣服并确认退货和尺码', 'stop_reason': 'validation_required',
        'proposal': {'kind': 'finish', 'answer': '保证合身免费退货，全部满足 https://fake',
                     'product_refs': [{'spu_id': 2, 'sku_id': 22, 'reason': '保证合身', 'evidence_ids': ['e']}]},
        'requirements': [{'id': 'query', 'text': '推荐棉质衣服并确认退货和尺码', 'source': 'user:query', 'status': 'satisfied', 'evidence_ids': ['e']}],
        'evidence': [{'run_id': 'r', 'evidence_id': 'e', 'tool': 'search_products', 'arguments': {}, 'status': 'ok', 'source': 'java',
                      'data': [{'spu_id': 2, 'sku_id': 22, 'sale_price': '299.90', 'available_stock': 2, 'material': 'cotton'}]}],
        'metrics': {'decisions': 2, 'tool_calls': 1, 'elapsed_ms': 30, 'token_usage': None},
    }


def test_dynamic_candidate_and_done_schema():
    result = validate(fixture(), current_product_refs=[{'spu_id': 1, 'sku_id': 11}])
    assert result['product_refs'][0]['sku_id'] == 22
    assert result['requirements'][0]['status'] == 'unconfirmed'
    assert '保证' not in result['answer'] + result['product_refs'][0]['reason']
    assert 'https://' not in result['answer']
    schema = json.loads((Path(__file__).parents[1] / 'contracts/assistant-streaming-chat/schemas/v2-pro-done.schema.json').read_text())
    jsonschema.validate(result, schema)


@pytest.mark.parametrize('mutation', ['fake', 'wrong_run', 'source', 'detail', 'missing_cite', 'zero_stock', 'over_budget', 'material', 'availability_mismatch'])
def test_invalid_or_unproven_refs_removed(mutation):
    value = fixture()
    evidence = value['evidence'][0]
    if mutation == 'fake': value['proposal']['product_refs'][0]['sku_id'] = 99
    if mutation == 'wrong_run': evidence['run_id'] = 'other'
    if mutation == 'source': evidence['source'] = 'rag'
    if mutation == 'detail': evidence['tool'] = 'get_product_detail'
    if mutation == 'missing_cite': value['proposal']['product_refs'][0]['evidence_ids'] = ['invented']
    if mutation == 'zero_stock': evidence['data'][0]['available_stock'] = 0
    if mutation == 'over_budget': evidence['data'][0]['sale_price'] = '300.01'
    if mutation == 'material': evidence['data'][0]['material'] = 'polyester'
    if mutation == 'availability_mismatch':
        evidence.update(tool='check_availability', arguments={'spu_id': 99, 'color': 'red', 'size': 'M'}, data=evidence['data'][0])
    result = validate(value, explicit_filters={'budget_max': '300.00', 'material': 'cotton'})
    assert result['product_refs'] == []
    assert result['stop_reason'] == 'validation_failed'
    assert '保证' not in result['answer']


@pytest.mark.parametrize('key', ['sale_price', 'main_image_url', 'url'])
def test_model_commerce_fields_rejected(key):
    value = fixture()
    value['proposal']['product_refs'][0][key] = 'fake'
    result = validate(value)
    assert result['stop_reason'] == 'invalid_response'
    assert not result['product_refs']


def test_generic_size_is_only_disclaimer_and_policy_without_source_unconfirmed():
    value = fixture()
    value['proposal']['product_refs'][0].update(size_advice='肯定合身', basis='generic_rule', evidence_ids=['e', 's'])
    value['evidence'].append({'run_id': 'r', 'evidence_id': 's', 'tool': 'recommend_size', 'arguments': {'height_cm': 170, 'weight_kg': 60}, 'status': 'ok', 'source': 'size', 'data': {'basis': 'generic_rule', 'recommended_size': 'M'}})
    result = validate(value)
    assert result['product_refs'][0]['basis'] == 'generic_rule'
    assert '通用' in result['product_refs'][0]['size_advice']
    assert '保证' not in result['answer']
    assert result['requirements'][0]['status'] == 'unconfirmed'


@pytest.mark.parametrize('stop', ['cancelled', 'budget_exhausted', 'needs_input', 'dependency_unavailable'])
def test_stopped_runs_preserve_uncertainty(stop):
    value = fixture()
    value.update(stop_reason=stop, proposal=None)
    result = validate(value)
    assert result['stop_reason'] == stop
    assert not result['product_refs']
    assert result['requirements'][0]['status'] in {'unconfirmed', 'needs_input'}


def test_context_binding_required():
    with pytest.raises(ValueError): validate(fixture(), run_id='foreign')


def test_latest_stock_observation_invalidates_earlier_citation():
    value = fixture()
    later = copy.deepcopy(value['evidence'][0])
    later['evidence_id'] = 'later'
    later['data'][0]['available_stock'] = 0
    value['evidence'].append(later)
    assert validate(value)['product_refs'] == []


def test_enforced_search_filter_codes_are_not_display_names():
    value = fixture()
    value['evidence'][0]['arguments'] = {'fit': 'regular', 'season': 'spring'}
    value['evidence'][0]['data'][0]['fit_type'] = '合身'
    result = validate(value, explicit_filters={'fit': 'regular', 'season': 'spring'})
    assert len(result['product_refs']) == 1


@pytest.mark.parametrize('tool,source,data', [('search_policy', 'policy', {'policy_answer': '保证退款'}), ('search_product_knowledge', 'rag', {'retrieved_chunks': [{'text': 'cotton'}]})])
def test_unrelated_local_evidence_cannot_support_product_ref(tool, source, data):
    value = fixture()
    value['evidence'].append({'run_id': 'r', 'evidence_id': 'local', 'tool': tool, 'source': source, 'status': 'ok', 'arguments': {'query': 'q'}, 'data': data})
    value['proposal']['product_refs'][0]['evidence_ids'].append('local')
    assert not validate(value)['product_refs']


def test_valid_policy_source_supplements_pair_without_guarantee():
    value = fixture()
    value['evidence'].append({'run_id': 'r', 'evidence_id': 'p', 'tool': 'search_policy', 'source': 'policy', 'status': 'ok', 'arguments': {'query': 'returns'}, 'data': {'policy_chunks': [{'text': 'Seven days', 'source': 'policy.md'}]}})
    value['proposal']['product_refs'][0]['evidence_ids'].append('p')
    assert validate(value)['product_refs']


def test_size_advice_without_size_evidence_does_not_invent_basis():
    value = fixture()
    value['proposal']['product_refs'][0].update(size_advice='M', basis='generic_rule')
    ref = validate(value)['product_refs'][0]
    assert ref.get('basis') is None


def test_generic_size_uses_observed_recommendation():
    value = fixture()
    value['proposal']['product_refs'][0].update(size_advice='XXL', basis='generic_rule', evidence_ids=['e', 's'])
    value['evidence'].append({'run_id': 'r', 'evidence_id': 's', 'tool': 'recommend_size', 'arguments': {'height_cm': 170, 'weight_kg': 60}, 'status': 'ok', 'source': 'size', 'data': {'basis': 'generic_rule', 'recommended_size': 'M'}})
    advice = validate(value)['product_refs'][0]['size_advice']
    assert 'M' in advice and 'XXL' not in advice


def test_needs_input_reports_missing_fields_from_observation():
    value = fixture()
    value.update(stop_reason='needs_input', proposal=None, observations=[{'status': 'needs_input', 'missing_fields': ['height_cm', 'weight_kg']}])
    assert 'height_cm' in validate(value)['answer']


@pytest.mark.parametrize('price,allowed', [('299.90', True), ('300.00', True), ('300.01', False)])
def test_exact_decimal_budget_boundary(price, allowed):
    value = fixture()
    value['evidence'][0]['data'][0]['sale_price'] = price
    assert bool(validate(value, explicit_filters={'budget_max': '300.00'})['product_refs']) is allowed


def test_validator_accepts_actual_executor_result():
    from clothing_assistant.agent.pro.executor import run_pro_agent
    from clothing_assistant.infrastructure.java_tool_client import tool_result
    value = fixture()
    actions = iter([{'kind': 'tool', 'tool': 'search_products', 'arguments': {}}, value['proposal']])
    result = run_pro_agent('推荐', decide=lambda snapshot, **kw: next(actions),
                          execute_tool=lambda action, **kw: tool_result('java', 'ok', value['evidence'][0]['data'], evidence_id='e'),
                          context={'run_id': 'r'})
    assert validate(result)['product_refs'][0]['sku_id'] == 22
