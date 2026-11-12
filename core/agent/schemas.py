"""Typed schemas for the automatic model-annotation agent.

Pydantic validation establishes structural well-formedness only. It does **not**
establish biological correctness. Semantic checks (target existence, dependency
validity, evidence consistency) are applied separately in ``core.agent.validate``
and still do not certify biological identity.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Reuse the strict grounded selection schema from the public reaction path.
from core.reaction.grounded import GroundedAnnotation


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EntityRepresentation(str, Enum):
    SBML_SPECIES = "sbml_species"
    FBC_GENE_PRODUCT = "fbc_gene_product"
    SBML_QUAL_SPECIES = "sbml_qual_species"
    SBML_REACTION = "sbml_reaction"
    SBML_QUAL_TRANSITION = "sbml_qual_transition"
    UNKNOWN = "unknown"


class BiologicalType(str, Enum):
    CHEMICAL = "chemical"
    GENE = "gene"
    PROTEIN = "protein"
    COMPLEX = "complex"
    REACTION = "reaction"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


class AnnotationStatus(str, Enum):
    MISSING = "missing"
    PARTIAL = "partial"
    COMPLETE = "complete"
    CONFLICTING = "conflicting"
    UNSUPPORTED = "unsupported"


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    UNRESOLVED = "unresolved"
    FAILED = "failed"
    SKIPPED = "skipped"


class FailureKind(str, Enum):
    UNCERTAIN_TYPE = "uncertain_type"
    UNSUPPORTED_IDENTITY = "unsupported_identity"
    UNRESOLVED_IDENTITY = "unresolved_identity"
    BIOLOGICAL_ABSTENTION = "biological_abstention"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    TECHNICAL_FAILURE = "technical_failure"
    NONE = "none"


class AnnotationScope(str, Enum):
    ADD_MISSING = "add_missing"
    REPLACE_EXISTING = "replace_existing"


ALLOWED_TOOLS = (
    "inspect_model",
    "classify_or_route_entities",
    "annotate_species_candidates",
    "assess_identity_support",
    "predict_qualifiers",
    "retrieve_reaction_rrf",
    "select_reaction_grounded",
    "write_species_annotations",
    "write_reaction_annotations",
    "persist_checkpoint",
    "export_reports",
)

ToolName = Literal[
    "inspect_model",
    "classify_or_route_entities",
    "annotate_species_candidates",
    "assess_identity_support",
    "predict_qualifiers",
    "retrieve_reaction_rrf",
    "select_reaction_grounded",
    "write_species_annotations",
    "write_reaction_annotations",
    "persist_checkpoint",
    "export_reports",
]

TaskToolName = Literal[
    "annotate_species_candidates",
    "assess_identity_support",
    "predict_qualifiers",
    "retrieve_reaction_rrf",
    "select_reaction_grounded",
    "write_species_annotations",
    "write_reaction_annotations",
]


# ---------------------------------------------------------------------------
# Inventory / entities
# ---------------------------------------------------------------------------


class ExistingAnnotation(StrictModel):
    database: str
    identifier: str
    qualifier: str = "is"
    uri: str = ""


class EntityRecord(StrictModel):
    target_id: str
    representation: EntityRepresentation
    biological_type: BiologicalType = BiologicalType.UNKNOWN
    display_name: str = ""
    annotation_status: AnnotationStatus = AnnotationStatus.MISSING
    existing_annotations: list[ExistingAnnotation] = Field(default_factory=list)
    participant_of: list[str] = Field(default_factory=list)
    participants: list[str] = Field(default_factory=list)
    support_notes: list[str] = Field(default_factory=list)


class ModelInventory(StrictModel):
    model_file: str
    model_id: str = ""
    model_format: str = "SBML"
    parse_ok: bool = True
    parse_error: Optional[str] = None
    entities: list[EntityRecord] = Field(default_factory=list)
    unsupported: list[EntityRecord] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    sha256: str = ""


# ---------------------------------------------------------------------------
# Typed tool arguments (per allowlisted tool)
# ---------------------------------------------------------------------------


class EmptyToolArgs(StrictModel):
    """Tools that take no tool-specific arguments beyond target/task ids."""


class AnnotateSpeciesArgs(StrictModel):
    databases: list[str] = Field(default_factory=lambda: ["chebi"])
    method: Literal["direct", "rag"] = "direct"
    predict_qualifiers: bool = False


class AssessIdentityArgs(StrictModel):
    effort: Literal["low", "medium", "high"] = "low"


class PredictQualifiersArgs(StrictModel):
    effort: Literal["low", "medium", "high"] = "low"


class RetrieveReactionArgs(StrictModel):
    allow_names_only: bool = True
    top_k: int = Field(default=100, ge=1, le=100)


class SelectReactionArgs(StrictModel):
    llm_model: str = "metabolm-local"
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "low"
    max_output_tokens: int = Field(default=512, ge=1, le=2048)


class WriteSpeciesArgs(StrictModel):
    add_all: bool = False
    qualifier: str = "is"


class WriteReactionArgs(StrictModel):
    require_selected: bool = True


TOOL_ARG_MODELS: dict[str, type[StrictModel]] = {
    "inspect_model": EmptyToolArgs,
    "classify_or_route_entities": EmptyToolArgs,
    "annotate_species_candidates": AnnotateSpeciesArgs,
    "assess_identity_support": AssessIdentityArgs,
    "predict_qualifiers": PredictQualifiersArgs,
    "retrieve_reaction_rrf": RetrieveReactionArgs,
    "select_reaction_grounded": SelectReactionArgs,
    "write_species_annotations": WriteSpeciesArgs,
    "write_reaction_annotations": WriteReactionArgs,
    "persist_checkpoint": EmptyToolArgs,
    "export_reports": EmptyToolArgs,
}


def parse_tool_arguments(tool: str, arguments: dict[str, Any] | StrictModel | None) -> StrictModel:
    model_cls = TOOL_ARG_MODELS.get(tool)
    if model_cls is None:
        raise ValueError(f"no argument schema for tool: {tool}")
    if arguments is None:
        return model_cls()
    if isinstance(arguments, StrictModel):
        return model_cls.model_validate(arguments.model_dump(mode="python"))
    return model_cls.model_validate(arguments)


# ---------------------------------------------------------------------------
# Typed tool results / evidence payloads
# ---------------------------------------------------------------------------


class SpeciesRecommendationRow(StrictModel):
    model_config = ConfigDict(extra="allow")

    id: str
    display_name: str = ""
    annotation: str = ""
    database: str = ""
    type: str = ""
    qualifier: str = "is"
    update_annotation: str = "ignore"
    identity_status: str = ""
    status: str = ""
    Note: str = ""
    match_score: Optional[float] = None
    component_id: str = ""
    qualifier_status: str = ""


class ReactionRecommendationRow(StrictModel):
    model_config = ConfigDict(extra="allow")

    id: str
    display_name: str = ""
    annotation: str = ""
    database: str = "kegg"
    type: str = "reaction"
    qualifier: str = "is"
    update_annotation: str = "ignore"
    status: str = ""
    biological_abstain: Optional[bool] = None
    error: Optional[str] = None
    match_score: Optional[float] = None
    reasoning_summary: str = ""


class InspectModelResult(StrictModel):
    inventory: ModelInventory


class RouteEntry(StrictModel):
    target_id: str
    representation: EntityRepresentation
    biological_type: BiologicalType
    annotation_status: AnnotationStatus
    supported: bool


class ClassifyRouteResult(StrictModel):
    routing: list[RouteEntry] = Field(default_factory=list)


class AnnotateSpeciesResult(StrictModel):
    n_rows: int = 0
    unresolved_ids: list[str] = Field(default_factory=list)
    recommendations: list[SpeciesRecommendationRow] = Field(default_factory=list)


class IdentityAssessment(StrictModel):
    id: str
    identity_status: str


class AssessIdentityResult(StrictModel):
    assessments: list[IdentityAssessment] = Field(default_factory=list)


class QualifierPrediction(StrictModel):
    id: str
    qualifier: str = "is"
    qualifier_status: str = "ok"


class PredictQualifiersResult(StrictModel):
    qualifiers: list[QualifierPrediction] = Field(default_factory=list)


class RetrievalArtifact(StrictModel):
    """RRF retrieval artifact for one reaction; evidence rows stay opaque dicts

    until checked with ``core.reaction.grounded.validate_evidence``.
    """

    reaction_id: str
    query: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    display_name: str = ""


class RetrieveReactionResult(StrictModel):
    n_retrieved: int = 0
    reaction_ids: list[str] = Field(default_factory=list)


class RetrieveReactionEvidence(StrictModel):
    artifacts: list[RetrievalArtifact] = Field(default_factory=list)


class SelectReactionResult(StrictModel):
    selected_ids: list[str] = Field(default_factory=list)
    abstained_ids: list[str] = Field(default_factory=list)
    failed_ids: list[str] = Field(default_factory=list)
    recommendations: list[ReactionRecommendationRow] = Field(default_factory=list)


class WritePathResult(StrictModel):
    path: str


class CheckpointResult(StrictModel):
    path: str


class ExportQueuedResult(StrictModel):
    queued: bool = True


TOOL_RESULT_MODELS: dict[str, type[StrictModel]] = {
    "inspect_model": ModelInventory,  # tools currently dump inventory dict
    "classify_or_route_entities": ClassifyRouteResult,
    "annotate_species_candidates": AnnotateSpeciesResult,
    "assess_identity_support": AssessIdentityResult,
    "predict_qualifiers": PredictQualifiersResult,
    "retrieve_reaction_rrf": RetrieveReactionResult,
    "select_reaction_grounded": SelectReactionResult,
    "write_species_annotations": WritePathResult,
    "write_reaction_annotations": WritePathResult,
    "persist_checkpoint": CheckpointResult,
    "export_reports": ExportQueuedResult,
}


class SemanticCheckReport(StrictModel):
    """Outcome of semantic checks. Passing does not imply biological correctness."""

    ok: bool
    problems: list[str] = Field(default_factory=list)
    checks_performed: list[str] = Field(default_factory=list)
    biological_correctness_certified: Literal[False] = False


# ---------------------------------------------------------------------------
# Plans / actions / observations / reports
# ---------------------------------------------------------------------------


class AnnotationTask(StrictModel):
    task_id: str
    target_id: str
    representation: EntityRepresentation
    biological_type: BiologicalType
    tool: TaskToolName
    databases: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    arguments: dict[str, Any] = Field(default_factory=dict)
    state: TaskState = TaskState.PENDING
    failure_kind: FailureKind = FailureKind.NONE
    reason: str = ""
    cache_identity: str = ""

    def parsed_arguments(self) -> StrictModel:
        return parse_tool_arguments(self.tool, self.arguments)


class AgentAction(StrictModel):
    action_id: str
    tool: ToolName
    target_ids: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""

    def parsed_arguments(self) -> StrictModel:
        return parse_tool_arguments(self.tool, self.arguments)


class AnnotationPlan(StrictModel):
    plan_version: int = 1
    scope: AnnotationScope = AnnotationScope.ADD_MISSING
    tasks: list[AnnotationTask] = Field(default_factory=list)
    actions: list[AgentAction] = Field(default_factory=list)
    limits: dict[str, Any] = Field(default_factory=dict)
    revision_reason: str = ""
    settings: dict[str, Any] = Field(default_factory=dict)
    semantic_checks: Optional[SemanticCheckReport] = None


class ToolObservation(StrictModel):
    action_id: str
    tool: ToolName
    ok: bool
    target_ids: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
    result: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    failure_kind: FailureKind = FailureKind.NONE
    evidence: dict[str, Any] = Field(default_factory=dict)
    semantic_checks: Optional[SemanticCheckReport] = None
    # Reused grounded answer when selection succeeds (schema-valid only).
    grounded_answer: Optional[GroundedAnnotation] = None


class EntityRuntimeState(StrictModel):
    target_id: str
    state: TaskState = TaskState.PENDING
    failure_kind: FailureKind = FailureKind.NONE
    identity: Optional[str] = None
    qualifier: str = "is"
    identity_status: str = ""
    tied_identities: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    last_observation: Optional[str] = None


class ExecutionLimits(StrictModel):
    max_actions: int = 40
    max_revisions: int = 6
    max_retries: int = 2
    max_runtime_sec: float = 600.0
    allow_names_only_reactions: bool = True


class PlannerDecision(StrictModel):
    """Structured LLM output for planning or revision."""

    decision: Literal["plan", "revise", "continue", "stop", "ask_user"]
    actions: list[AgentAction] = Field(default_factory=list)
    tasks: list[AnnotationTask] = Field(default_factory=list)
    reason: str = ""
    ask_user_prompt: str = ""


class AutoAnnotateReport(StrictModel):
    status: Literal["completed", "partial", "failed", "interrupted"]
    annotated_model_path: Optional[str] = None
    change_table_path: Optional[str] = None
    initial_plan_path: Optional[str] = None
    executed_plan_path: Optional[str] = None
    recommendations_path: Optional[str] = None
    provenance_path: Optional[str] = None
    completed_ids: list[str] = Field(default_factory=list)
    unresolved_ids: list[str] = Field(default_factory=list)
    skipped_ids: list[str] = Field(default_factory=list)
    failed_ids: list[str] = Field(default_factory=list)
    technical_errors: list[str] = Field(default_factory=list)
    biological_reservations: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    partial: bool = False
    resume_info: dict[str, Any] = Field(default_factory=dict)
    schema_validated: bool = True
    biological_correctness_certified: Literal[False] = False


class AutoAnnotateResult(StrictModel):
    report: AutoAnnotateReport
    inventory: Optional[ModelInventory] = None
    initial_plan: Optional[AnnotationPlan] = None
    executed_plan: Optional[AnnotationPlan] = None
    entity_states: dict[str, EntityRuntimeState] = Field(default_factory=dict)
    observations: list[ToolObservation] = Field(default_factory=list)
    output_dir: str = ""
    provenance: dict[str, Any] = Field(default_factory=dict)
