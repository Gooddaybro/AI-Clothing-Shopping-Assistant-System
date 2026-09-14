"""Strict model actions defined by the shared assistant v2 contract."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter


Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Identifier = Annotated[Text, Field(max_length=128)]
EvidenceIds = Annotated[list[Identifier], Field(max_length=20)]
PositiveId = Annotated[int, Field(gt=0)]
Measurement = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Money = Annotated[str, StringConstraints(pattern=r'^(0|[1-9][0-9]*)(\.[0-9]{1,2})?$')]
RequirementStatus = Literal['pending', 'satisfied', 'needs_input', 'unconfirmed']


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)


class RequirementUpdate(StrictModel):
    requirement_id: Identifier
    status: RequirementStatus
    evidence_ids: EvidenceIds


class Requirement(StrictModel):
    id: Identifier
    text: Annotated[Text, Field(max_length=500)]
    source: Text
    status: RequirementStatus = 'pending'
    evidence_ids: EvidenceIds = Field(default_factory=list)


class ProductRef(StrictModel):
    spu_id: PositiveId
    sku_id: PositiveId
    reason: Text
    size_advice: Text | None = None
    basis: Literal['generic_rule', 'product_chart'] | None = None
    evidence_ids: EvidenceIds = Field(default_factory=list)


class SearchProductsArguments(StrictModel):
    category: str | None = None
    style: str | None = None
    season: str | None = None
    material: str | None = None
    fit: str | None = None
    gender: str | None = None
    recall_text: str | None = None
    budget_max: Money | None = None


class ProductDetailArguments(StrictModel):
    spu_id: PositiveId


class AvailabilityArguments(ProductDetailArguments):
    color: Text
    size: Text


class PolicyArguments(StrictModel):
    query: Text


class ProductKnowledgeArguments(PolicyArguments):
    spu_id: PositiveId | None = None


class SizeArguments(StrictModel):
    height_cm: Measurement
    weight_kg: Measurement
    preferred_fit: str | None = None
    spu_id: PositiveId | None = None


class ActionBase(StrictModel):
    requirement_updates: list[RequirementUpdate] = Field(default_factory=list, max_length=12)


class ToolActionBase(ActionBase):
    kind: Literal['tool']


class SearchProductsAction(ToolActionBase):
    tool: Literal['search_products']
    arguments: SearchProductsArguments


class ProductDetailAction(ToolActionBase):
    tool: Literal['get_product_detail']
    arguments: ProductDetailArguments


class AvailabilityAction(ToolActionBase):
    tool: Literal['check_availability']
    arguments: AvailabilityArguments


class ProductKnowledgeAction(ToolActionBase):
    tool: Literal['search_product_knowledge']
    arguments: ProductKnowledgeArguments


class PolicyAction(ToolActionBase):
    tool: Literal['search_policy']
    arguments: PolicyArguments


class SizeAction(ToolActionBase):
    tool: Literal['recommend_size']
    arguments: SizeArguments


ToolAction = Annotated[
    SearchProductsAction | ProductDetailAction | AvailabilityAction
    | ProductKnowledgeAction | PolicyAction | SizeAction,
    Field(discriminator='tool'),
]


class ClarifyAction(ActionBase):
    kind: Literal['clarify']
    question: Text


class FinishAction(ActionBase):
    kind: Literal['finish']
    answer: Annotated[Text, Field(max_length=8000)]
    product_refs: list[ProductRef] = Field(max_length=20)


Action = Annotated[ToolAction | ClarifyAction | FinishAction, Field(discriminator='kind')]
_ACTION_ADAPTER = TypeAdapter(Action)


def parse_action(payload: object) -> Action:
    """Validate one decoded JSON object without coercing model-generated values."""
    return _ACTION_ADAPTER.validate_python(payload)
