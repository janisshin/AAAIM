"""Typed schemas for the automatic model-annotation agent.

Pydantic validation establishes structural well-formedness only. It does **not**
establish biological correctness. Semantic checks (target existence, dependency
validity, evidence consistency) are applied separately in ``core.agent.validate``
and still do not certify biological identity.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    annotator_kind: Optional[str] = None


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
    annotator_kind: Optional[str] = None


class IdentityAssessment(StrictModel):
    id: str
    identity_status: str


class AssessIdentityResult(StrictModel):
    assessments: list[IdentityAssessment] = Field(default_factory=list)
    mode: Optional[str] = None


class QualifierPrediction(StrictModel):
    id: str
    qualifier: str = "is"
    qualifier_status: str = "ok"


class PredictQualifiersResult(StrictModel):
    qualifiers: list[QualifierPrediction] = Field(default_factory=list)
    mode: Optional[str] = None


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
    """Runtime budgets for the annotation controller.

    ``max_planning_turns`` counts ordinary continue/re-plan turns after action
    groups. ``max_failure_revisions`` counts failure-driven replans only.
    ``planner_max_new_tokens`` is configurable and justified by measured fit;
    512 is not a mandatory ceiling.
    """

    max_actions: int = 40
    max_planning_turns: int = 12
    max_failure_revisions: int = 6
    max_retries: int = 2
    max_runtime_sec: float = 600.0
    allow_names_only_reactions: bool = True
    max_actions_per_turn: int = 3
    planner_max_new_tokens: int = 768
    # Planner-only paid spend controls (default: zero paid dispatches).
    planner_spend_authorized: bool = False
    planner_spend_cap_usd: float = 0.0
    # Biological annotation model ids permitted in tool arguments.
    authorized_annotation_models: list[str] = Field(
        default_factory=lambda: ["metabolm-local"]
    )

    @classmethod
    def model_validate(cls, obj: Any, *args: Any, **kwargs: Any) -> "ExecutionLimits":
        # Backward-compatible alias used by older tests/callers.
        if isinstance(obj, dict) and "max_revisions" in obj and "max_failure_revisions" not in obj:
            obj = {**obj, "max_failure_revisions": obj.pop("max_revisions")}
        elif isinstance(obj, dict) and "max_revisions" in obj:
            obj = {k: v for k, v in obj.items() if k != "max_revisions"}
        return super().model_validate(obj, *args, **kwargs)

    def __init__(self, **data: Any) -> None:
        if "max_revisions" in data and "max_failure_revisions" not in data:
            data["max_failure_revisions"] = data.pop("max_revisions")
        elif "max_revisions" in data:
            data.pop("max_revisions")
        super().__init__(**data)

    @property
    def max_revisions(self) -> int:
        """Deprecated alias for ``max_failure_revisions``."""
        return self.max_failure_revisions


PlannerDecisionName = Literal["continue", "finish", "needs_clarification"]


class PlannerDecision(StrictModel):
    """Structured LLM output for one planning turn.

    - ``continue``: emit the next bounded action group (does not complete the run).
    - ``finish``: claim every in-scope target is terminal; controller validates the ledger.
    - ``needs_clarification``: stop for user input without inventing completion.

    Legacy aliases ``plan``/``revise``→continue, ``stop``→finish, ``ask_user``→
    needs_clarification are accepted on input.
    """

    decision: PlannerDecisionName
    actions: list[AgentAction] = Field(default_factory=list)
    tasks: list[AnnotationTask] = Field(default_factory=list)
    reason: str = ""
    clarification_prompt: str = ""
    # Legacy alias accepted on input via normalize_planner_decision().
    ask_user_prompt: str = ""

    @field_validator("decision", mode="before")
    @classmethod
    def _normalize_decision_name(cls, value: Any) -> Any:
        mapping = {
            "plan": "continue",
            "revise": "continue",
            "continue": "continue",
            "stop": "finish",
            "finish": "finish",
            "ask_user": "needs_clarification",
            "needs_clarification": "needs_clarification",
        }
        if isinstance(value, str) and value in mapping:
            return mapping[value]
        return value


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


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------

# v3: dual planner/inference identities + durable planner-only spend ledger.
# v2 checkpoints are rejected on resume (restart without resume_from).
CHECKPOINT_SCHEMA_VERSION: Literal[3] = 3
# Bump when allowlisted tool names or TOOL_ARG_MODELS / TOOL_RESULT_MODELS change.
TOOL_SCHEMA_VERSION: Literal["agent-tools-v1"] = "agent-tools-v1"


class CheckpointCompatibilityError(ValueError):
    """Checkpoint is missing, corrupt, unsupported, or incompatible with this run."""


class InFlightAction(StrictModel):
    """Action started but not yet durably completed (at-least-once on resume)."""

    action: AgentAction
    fingerprint: str


class CheckpointProvenance(StrictModel):
    planner_meta: list[dict[str, Any]] = Field(default_factory=list)
    revisions: list[dict[str, Any]] = Field(default_factory=list)
    stop_reason: Optional[str] = None
    inference_stages: list[Any] = Field(default_factory=list)
    species_annotator: dict[str, Any] = Field(default_factory=dict)
    last_planner_decision: Optional[str] = None
    clarification_prompt: str = ""
    execution_mode: Optional[str] = None
    llm_planning_succeeded: Optional[bool] = None
    planner_prompt_version: Optional[str] = None
    planner_schema_version: Optional[str] = None


class AgentCheckpoint(StrictModel):
    """Versioned durable controller state for interruption/resume.

    Incremental planning fields (``planning_turn_count``,
    ``failure_revision_count``, ``last_planner_decision``) are required for
    multi-turn continue/finish loops. ``revision_count`` remains as a synonym
    of ``failure_revision_count`` for resume-plan compatibility.

    schema_version=3 requires dual planner/inference identities and a planner
    budget ledger. Older checkpoints cannot be resumed.
    """

    schema_version: Literal[3] = CHECKPOINT_SCHEMA_VERSION
    cache_identity: str
    model_sha256: str
    backend: dict[str, Any] = Field(default_factory=dict)
    planner: dict[str, Any] = Field(default_factory=dict)
    inference: dict[str, Any] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)
    tool_schema_version: str = TOOL_SCHEMA_VERSION
    resource_identity: dict[str, Any] = Field(default_factory=dict)
    planner_budget: dict[str, Any] = Field(default_factory=dict)
    initial_plan: Optional[AnnotationPlan] = None
    executed_plan: Optional[AnnotationPlan] = None
    entity_states: dict[str, EntityRuntimeState] = Field(default_factory=dict)
    observations: list[ToolObservation] = Field(default_factory=list)
    completed_fingerprints: list[str] = Field(default_factory=list)
    action_counts: dict[str, int] = Field(default_factory=dict)
    actions_executed: int = 0
    revision_count: int = 0  # failure-driven revisions (compat alias)
    planning_turn_count: int = 0
    failure_revision_count: int = 0
    last_planner_decision: Optional[PlannerDecisionName] = None
    species_recommendations: Optional[list[dict[str, Any]]] = None
    reaction_recommendations: Optional[list[dict[str, Any]]] = None
    reaction_evidence: list[dict[str, Any]] = Field(default_factory=list)
    entity_identities: dict[str, dict[str, Any]] = Field(default_factory=dict)
    paths: dict[str, Any] = Field(default_factory=dict)
    provenance: CheckpointProvenance = Field(default_factory=CheckpointProvenance)
    in_flight_action: Optional[InFlightAction] = None
