"""Conservative internal-to-v2 validation. Java must still re-read live facts.

Only executor-owned evidence and Java-assembled context belong at this boundary.
Free-form model claims are never published: Task6 tracks the whole query, so it
cannot prove each clause of a compound request or certify policy/fit guarantees.
"""

import re
from collections.abc import Mapping
from decimal import Decimal

from .schemas import FinishAction


_FILTER_FIELDS = {'category': 'category', 'material': 'material', 'fit': 'fit_type',
                  'style': 'style_tags', 'season': 'season', 'gender': 'gender'}
_SOURCES = {'search_products': 'java', 'check_availability': 'java',
            'get_product_detail': 'java', 'search_policy': 'policy',
            'search_product_knowledge': 'rag', 'recommend_size': 'size'}
_GENERIC = '仅有通用身高体重规则参考，未核验目标商品尺码表，无法确认合身。'


def _money(value):
    if not isinstance(value, str) or not re.fullmatch(r'(0|[1-9][0-9]*)(\.[0-9]{1,2})?', value):
        return None
    return Decimal(value)


def _pair(row):
    values = (row.get('spu_id'), row.get('sku_id'))
    return values if all(type(v) is int and v > 0 for v in values) else None


def _rows(evidence):
    tool, data = evidence.get('tool'), evidence.get('data')
    if tool == 'search_products' and isinstance(data, list):
        return [row for row in data if isinstance(row, Mapping) and _pair(row)]
    if tool == 'check_availability' and isinstance(data, Mapping) and _pair(data):
        args = evidence.get('arguments', {})
        if all(args.get(k) is not None and args[k] == data.get(k) for k in ('spu_id', 'color', 'size')):
            return [data]
    return []


def _local_source(evidence):
    data = evidence.get('data')
    if not isinstance(data, Mapping):
        return False
    key = {'search_policy': 'policy_chunks', 'search_product_knowledge': 'retrieved_chunks'}.get(evidence.get('tool'))
    chunks = data.get(key) if key else None
    return isinstance(chunks, list) and bool(chunks) and all(
        isinstance(chunk, Mapping) and any(isinstance(chunk.get(k), str) and chunk[k].strip()
                                         for k in ('source', 'source_file', 'document_id', 'chunk_id', 'evidence_id'))
        for chunk in chunks
    )


def _matches(row, filters, searches):
    stock = row.get('available_stock')
    if type(stock) is not int or stock <= 0 or _money(row.get('sale_price')) is None:
        return False
    for key, expected in filters.items():
        if expected is None:
            continue
        if key == 'budget_max':
            bound = _money(expected)
            if bound is None or _money(row['sale_price']) > bound:
                return False
        elif key in _FILTER_FIELDS:
            # Java filters use codes; returned fields may be localized labels.
            if any(e.get('arguments', {}).get(key) == expected for e in searches):
                continue
            actual = row.get(_FILTER_FIELDS[key])
            if actual != expected and not (isinstance(actual, list) and expected in actual):
                return False
        else:
            return False
    return True


def validate_pro_result(result, *, context):
    """Return a v2 done candidate, never a public/Java approval.

    IDs must be supplied by trusted request context and run_id must match the
    executor. Evidence is local to that executor run; initial candidates alone
    cannot authorize references. Contradictory or unverifiable prose is replaced
    deterministically, retaining valid partial product references and uncertainty.
    """
    ids = {key: context.get(key) for key in ('request_id', 'thread_id', 'run_id')}
    if any(not isinstance(v, str) or not v.strip() for v in ids.values()) or ids['run_id'] != result.get('run_id'):
        raise ValueError('Trusted request IDs and matching executor run_id are required')
    filters = context.get('explicit_filters', {})
    evidence = {}
    duplicates = set()
    for item in result.get('evidence', []):
        eid = item.get('evidence_id')
        if not isinstance(eid, str) or not eid.strip() or len(eid) > 128:
            continue
        if eid in evidence:
            duplicates.add(eid)
        if item.get('run_id') == ids['run_id'] and item.get('status') == 'ok' and _SOURCES.get(item.get('tool')) == item.get('source'):
            evidence[eid] = item
    for eid in duplicates:
        evidence.pop(eid, None)
    latest = {}
    searches = {}
    for item in evidence.values():
        for row in _rows(item):
            pair = _pair(row)
            latest[pair] = {**latest.get(pair, {}), **row}
            if item.get('tool') == 'search_products':
                searches.setdefault(pair, []).append(item)
    stop = result.get('stop_reason')
    refs = []
    proposal = None
    if stop == 'validation_required':
        try:
            proposal = FinishAction.model_validate(result.get('proposal'))
        except (ValueError, TypeError):
            stop = 'invalid_response'
    if proposal is not None:
        for ref in proposal.product_refs:
            pair = (ref.spu_id, ref.sku_id)
            cited = [evidence[eid] for eid in ref.evidence_ids if eid in evidence]
            pair_evidence = [e for e in cited if any(_pair(row) == pair for row in _rows(e))]
            # A local citation may supplement a pair but can never authorize it.
            relevant = all(
                any(_pair(row) == pair for row in _rows(e))
                or (e.get('tool') == 'get_product_detail' and e.get('arguments', {}).get('spu_id') == ref.spu_id
                    and isinstance(e.get('data'), Mapping) and e['data'].get('spu_id') == ref.spu_id)
                or _local_source(e)
                or (e.get('tool') == 'recommend_size' and e.get('arguments', {}).get('spu_id') in (None, ref.spu_id))
                for e in cited
            )
            if len(cited) != len(ref.evidence_ids) or not relevant or not pair_evidence or not _matches(latest.get(pair, {}), filters, searches.get(pair, [])):
                continue
            safe = {'spu_id': ref.spu_id, 'sku_id': ref.sku_id,
                    'reason': '本次商品查询已返回该商品，价格和库存仍需商城最终核验。',
                    'evidence_ids': [e['evidence_id'] for e in pair_evidence]}
            if ref.size_advice is not None or ref.basis is not None:
                size = next((e for e in reversed(cited) if e.get('tool') == 'recommend_size'
                             and isinstance(e.get('data'), Mapping)
                             and isinstance(e['data'].get('recommended_size'), str)
                             and re.fullmatch(r'[A-Za-z0-9 /.-]{1,20}', e['data']['recommended_size'])), None)
                if size is not None:
                    data = size['data']
                    chart = (data.get('basis') == 'product_chart'
                             and data.get('product_chart_available') is True
                             and data.get('product_chart_evidence') is True
                             and size.get('arguments', {}).get('spu_id') == ref.spu_id)
                    limitation = '目标商品尺码表参考，实际穿着效果仍需确认。' if chart else _GENERIC
                    safe.update(size_advice=data['recommended_size'] + '：' + limitation,
                                basis='product_chart' if chart else 'generic_rule')
                    safe['evidence_ids'].append(size['evidence_id'])
            if not any((r['spu_id'], r['sku_id']) == pair for r in refs):
                refs.append(safe)
        stop = 'validation_failed' if len(refs) != len(proposal.product_refs) else 'completed'
    allowed = {'completed', 'needs_input', 'budget_exhausted', 'validation_failed', 'dependency_unavailable', 'invalid_response', 'cancelled'}
    if stop not in allowed:
        stop = 'invalid_response'
    # Rebuild source text from trusted input, not model status suggestions.
    requirements = [{'id': 'query', 'text': str(result.get('query') or '用户请求')[:500],
                     'status': 'needs_input' if stop == 'needs_input' else 'unconfirmed', 'evidence_ids': []}]
    for key, value in filters.items():
        if value is None:
            continue
        requirements.append({'id': key, 'text': f'{key}: {value}'[:500],
                             'status': 'unconfirmed', 'evidence_ids': []})
    answer = ('已保留本次查询中有依据的商品供参考。' if refs else '当前没有可确认的商品推荐。')
    answer += '尚未确认请求中的全部条件；退换政策、商品适配和尺码需相应证据核验，合身程度尚待确认。'
    if stop == 'needs_input':
        missing = []
        for observation in result.get('observations', []):
            if observation.get('status') == 'needs_input':
                missing = [field for field in observation.get('missing_fields', [])
                           if isinstance(field, str) and re.fullmatch(r'[a-z_]{1,40}', field)]
        answer += '请补充：' + '、'.join(missing) + '。' if missing else '请补充完成请求所需的信息。'
    elif stop in {'cancelled', 'budget_exhausted', 'dependency_unavailable'}:
        answer += '本次处理已停止，未完成的条件保持未确认。'
    return {'contract_version': 'assistant-v2', 'agent_mode': 'pro', **ids,
            'answer': answer, 'product_refs': refs, 'requirements': requirements,
            'stop_reason': stop, 'metrics': dict(result['metrics'])}

