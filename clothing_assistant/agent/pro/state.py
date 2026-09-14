"""Per-run collections and protected user constraints, without an executor."""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from .schemas import Requirement, RequirementUpdate


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def new_state(run_id: str, query: str, hard_constraints: Mapping[str, Any]) -> Mapping:
    """Keep immutable original user input alongside run-local mutable collections.

    Trusted initialization supplies explicit filters; model actions cannot replace
    them. Evidence linkage is supplied by trusted validation, never by an action.
    """
    sources = {key: f'user:explicit_filters.{key}' for key in hard_constraints}
    return MappingProxyType({
        'run_id': run_id,
        'query': query,
        'query_source': 'user:query',
        'hard_constraints': _freeze(hard_constraints),
        'hard_constraint_sources': _freeze(sources),
        'soft_constraints': {},
        'requirements': [
            Requirement(id=key, text=f'{key}: {value}', source=sources[key])
            for key, value in hard_constraints.items()
        ],
        'evidence': [],
    })


def apply_requirement_updates(state: Mapping, updates: Sequence) -> None:
    """Apply status suggestions only; satisfied requires usable, linked run evidence.

    Evidence records add run_id and requirement_ids to a tool observation after
    trusted validation has established relevance. Mere tool success is not proof
    that a budget, stock, policy or fit requirement has been satisfied.
    """
    parsed = [RequirementUpdate.model_validate(update) for update in updates]
    requirements = state['requirements']
    indexes = {requirement.id: index for index, requirement in enumerate(requirements)}
    if any(update.requirement_id not in indexes for update in parsed):
        raise ValueError('Unknown requirement ID')
    for update in parsed:
        if update.status == 'satisfied':
            usable_ids = {
                item.get('evidence_id') for item in state['evidence']
                if item.get('run_id') == state['run_id']
                and item.get('status') == 'ok'
                and update.requirement_id in item.get('requirement_ids', [])
            }
            if not update.evidence_ids or not set(update.evidence_ids) <= usable_ids:
                continue
        index = indexes[update.requirement_id]
        requirements[index] = requirements[index].model_copy(update={
            'status': update.status, 'evidence_ids': list(update.evidence_ids),
        })
