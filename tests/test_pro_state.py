import pytest
from pydantic import ValidationError

from clothing_assistant.agent.pro.schemas import parse_action
from clothing_assistant.agent.pro.state import apply_requirement_updates, new_state


@pytest.mark.parametrize(('tool', 'arguments'), [
    ('search_products', {'budget_max': '300.00', 'category': 'outerwear'}),
    ('get_product_detail', {'spu_id': 1}),
    ('check_availability', {'spu_id': 1, 'color': 'black', 'size': 'L'}),
    ('search_product_knowledge', {'query': 'material', 'spu_id': 1}),
    ('search_policy', {'query': 'returns'}),
    ('recommend_size', {'height_cm': 175, 'weight_kg': 70, 'preferred_fit': 'regular'}),
])
def test_six_tool_arguments_are_parsed(tool, arguments):
    action = parse_action({'kind': 'tool', 'tool': tool, 'arguments': arguments})
    assert action.tool == tool
    assert action.arguments.model_dump(exclude_none=True) == arguments
    with pytest.raises(ValidationError):
        parse_action({'kind': 'tool', 'tool': tool, 'arguments': {**arguments, 'user_id': 1}})


@pytest.mark.parametrize('payload', [
    {'kind': 'tool', 'tool': 'run_sql', 'arguments': {}},
    {'kind': 'tool', 'tool': 'search_products', 'arguments': {'url': 'evil'}},
    {'kind': 'tool', 'tool': 'search_products', 'arguments': {'budget_max': 300}},
    {'kind': 'tool', 'tool': 'get_product_detail', 'arguments': {'spu_id': True}},
    {'kind': 'tool', 'tool': 'get_product_detail', 'arguments': {'spu_id': '1'}},
    {'kind': 'tool', 'tool': 'check_availability', 'arguments': {'spu_id': 1, 'color': ' ', 'size': 'L'}},
    {'kind': 'tool', 'tool': 'recommend_size', 'arguments': {'height_cm': '175', 'weight_kg': 70}},
    {'kind': 'clarify', 'question': 'fit?', 'tool': 'search_policy'},
    {'kind': 'finish', 'answer': 'done', 'product_refs': [], 'arguments': {}},
    {'kind': 'finish', 'answer': 'done', 'product_refs': [{'spu_id': 1, 'sku_id': 2, 'reason': 'ok', 'price': '1.00'}]},
    {'kind': 'clarify', 'question': 'fit?', 'requirement_updates': [{'requirement_id': 'budget_max', 'status': 'satisfied', 'evidence_ids': [], 'source': 'model'}]},
])
def test_invalid_actions_are_rejected(payload):
    with pytest.raises(ValidationError):
        parse_action(payload)


def test_non_tool_actions():
    assert parse_action({'kind': 'clarify', 'question': 'fit?'}).question == 'fit?'
    assert parse_action({'kind': 'finish', 'answer': 'done', 'product_refs': []}).product_refs == []


def test_runs_do_not_share_collections():
    first = new_state('a', 'outerwear', {'budget_max': '300.00'})
    second = new_state('b', 'size', {})
    first['evidence'].append({'id': 'e1'})
    first['requirements'].clear()
    assert second['evidence'] == []
    assert first['requirements'] is not second['requirements']


def test_user_constraints_are_detached_and_immutable():
    constraints = {'budget_max': '300.00', 'colors': ['black']}
    state = new_state('a', 'outerwear', constraints)
    constraints['colors'].append('white')
    assert state['hard_constraints']['colors'] == ('black',)
    assert state['query'] == 'outerwear'
    assert state['query_source'] == 'user:query'
    assert state['hard_constraint_sources']['budget_max'] == 'user:explicit_filters.budget_max'
    with pytest.raises(TypeError):
        state['hard_constraints']['budget_max'] = '900.00'
    with pytest.raises(TypeError):
        state['hard_constraints'] = {}


@pytest.mark.parametrize('evidence', [
    [],
    [{'evidence_id': 'e1', 'run_id': 'other', 'status': 'ok', 'requirement_ids': ['budget_max']}],
    [{'evidence_id': 'e1', 'run_id': 'a', 'status': 'unavailable', 'requirement_ids': ['budget_max']}],
    [{'evidence_id': 'e1', 'run_id': 'a', 'status': 'empty', 'requirement_ids': ['budget_max']}],
    [{'evidence_id': 'e1', 'run_id': 'a', 'status': 'needs_input', 'requirement_ids': ['budget_max']}],
    [{'evidence_id': 'e1', 'run_id': 'a', 'status': 'no_evidence', 'requirement_ids': ['budget_max']}],
    [{'evidence_id': 'e1', 'run_id': 'a', 'status': 'forbidden', 'requirement_ids': ['budget_max']}],
    [{'evidence_id': 'e1', 'run_id': 'a', 'status': 'ok', 'requirement_ids': ['size']}],
])
def test_unusable_evidence_cannot_satisfy_requirement(evidence):
    state = new_state('a', 'outerwear', {'budget_max': '300.00'})
    state['evidence'].extend(evidence)
    apply_requirement_updates(state, [{'requirement_id': 'budget_max', 'status': 'satisfied', 'evidence_ids': ['e1']}])
    assert state['requirements'][0].status == 'pending'


def test_verified_update_preserves_original_requirement():
    state = new_state('a', 'outerwear', {'budget_max': '300.00'})
    state['evidence'].append({'evidence_id': 'e1', 'run_id': 'a', 'status': 'ok', 'requirement_ids': ['budget_max']})
    original = state['requirements'][0]
    action = parse_action({'kind': 'finish', 'answer': 'done', 'product_refs': [], 'requirement_updates': [{'requirement_id': 'budget_max', 'status': 'satisfied', 'evidence_ids': ['e1']}]})
    apply_requirement_updates(state, action.requirement_updates)
    result = state['requirements'][0]
    assert result.status == 'satisfied'
    assert (result.text, result.source) == (original.text, original.source)
    assert result.evidence_ids == ['e1']


def test_unknown_requirement_and_constraint_rewrite_are_rejected():
    state = new_state('a', 'outerwear', {'budget_max': '300.00'})
    with pytest.raises(ValueError):
        apply_requirement_updates(state, [{'requirement_id': 'invented', 'status': 'pending', 'evidence_ids': []}])
    with pytest.raises(ValidationError):
        apply_requirement_updates(state, [{'requirement_id': 'budget_max', 'status': 'pending', 'evidence_ids': [], 'hard_constraints': {}}])


@pytest.mark.parametrize('evidence_ids', [[], ['e1', 'missing']])
def test_satisfaction_requires_all_cited_evidence(evidence_ids):
    state = new_state('a', 'outerwear', {'budget_max': '300.00'})
    state['evidence'].append({'evidence_id': 'e1', 'run_id': 'a', 'status': 'ok', 'requirement_ids': ['budget_max']})
    apply_requirement_updates(state, [{'requirement_id': 'budget_max', 'status': 'satisfied', 'evidence_ids': evidence_ids}])
    assert state['requirements'][0].status == 'pending'


def test_model_can_report_missing_input_without_claiming_evidence():
    state = new_state('a', 'outerwear', {'budget_max': '300.00'})
    apply_requirement_updates(state, [{'requirement_id': 'budget_max', 'status': 'needs_input', 'evidence_ids': []}])
    assert state['requirements'][0].status == 'needs_input'


@pytest.mark.parametrize('amount', ['300', '299.9', '300.00'])
def test_decimal_money_matches_v2_request(amount):
    assert parse_action({'kind': 'tool', 'tool': 'search_products', 'arguments': {'budget_max': amount}}).arguments.budget_max == amount


@pytest.mark.parametrize('changes', [
    {'answer': 'a' * 8001},
    {'product_refs': [{'spu_id': 1, 'sku_id': 2, 'reason': 'ok'}] * 21},
    {'requirement_updates': [{'requirement_id': 'r', 'status': 'pending', 'evidence_ids': []}] * 13},
    {'requirement_updates': [{'requirement_id': 'r' * 129, 'status': 'pending', 'evidence_ids': []}]},
    {'requirement_updates': [{'requirement_id': 'r', 'status': 'pending', 'evidence_ids': ['e'] * 21}]},
    {'requirement_updates': [{'requirement_id': 'r', 'status': 'pending', 'evidence_ids': ['e' * 129]}]},
])
def test_actions_obey_result_contract_limits(changes):
    with pytest.raises(ValidationError):
        parse_action({'kind': 'finish', 'answer': 'done', 'product_refs': [], **changes})
