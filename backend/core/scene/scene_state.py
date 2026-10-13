"""
SmartClass AI — Universal Scene Representation (USR)
Defines the core data contracts for scene-level and entity-level states,
enabling domain-agnostic perception, spatial scene graphs, and evidence fusion.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any, Optional


@dataclass
class EntityState:
    """
    A tracked entity in the scene (person, object, or seat anchor).
    Decoupled from classroom-specific concepts while providing typed access
    for classroom domain operations.
    """
    entity_id: str                                  # e.g., "track_3", "seat_R1C1"
    entity_class: str = "person"                    # "person" | "object" | "furniture"

    # Geometry (image-space)
    bbox: List[int] = field(default_factory=lambda: [0, 0, 0, 0])      # [x1, y1, x2, y2]
    seat_anchor: Tuple[float, float] = (0.0, 0.0)   # (cx, y_bottom) — contact point
    depth: Optional[float] = None                   # estimated monocular depth

    # Pose & Gaze (for persons)
    pose_state: Optional[str] = None                # "seated" | "standing" | "leaning"
    head_direction: Optional[str] = None            # "FRONT" | "DOWN" | "LEFT" | "RIGHT"
    head_pose: Dict[str, float] = field(default_factory=lambda: {"yaw": 0.0, "pitch": 0.0, "roll": 0.0})
    gaze_confidence: float = 0.0

    # Activity & Attention
    activity: Optional[str] = None                  # "attentive" | "studying" | "phone" | etc.
    activity_confidence: float = 0.0
    attention_score: float = 0.0                    # 0.0 - 100.0 smoothed score
    attention_raw: float = 0.0
    attention_breakdown: Dict[str, float] = field(default_factory=dict)

    # Spatial assignment
    seat_id: Optional[int] = None
    seat_label: Optional[str] = None                # "R1C1"
    row: Optional[int] = None                       # 0-indexed row
    col: Optional[int] = None                       # 0-indexed column

    # Identity
    student_id: Optional[int] = None
    student_name: Optional[str] = None
    identity_confidence: float = 0.0

    # Confidence breakdown
    detection_confidence: float = 0.0
    tracking_confidence: float = 0.0
    association_confidence: float = 0.0

    # Status / Flagging
    status: str = "OK"                              # "OK" | "UNASSIGNED" | "GHOST"
    is_ghost: bool = False
    cost: float = 0.0

    # Peripherals
    phone_detected: bool = False
    laptop_detected: bool = False
    earphone_detected: bool = False

    # Extensible metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EntityState:
        # Filter only recognized keys
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class SceneRelationship:
    """A semantic or spatial relationship between two entities in the scene graph."""
    source_id: str                                  # e.g. "track_1"
    relation: str                                   # "adjacent_to", "same_row_as", "interacting_with"
    target_id: str                                  # e.g. "track_2"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SceneState:
    """
    Complete scene understanding snapshot at a point in time.
    Provides the unified state that bridges raw perception to reasoning engines.
    """
    timestamp: float = field(default_factory=time.time)
    frame_idx: int = 0
    entities: List[EntityState] = field(default_factory=list)

    # Scene-level aggregations
    people_count: int = 0
    occupied_seats: int = 0
    total_seats: int = 0
    average_attention: float = 0.0

    # Topology snapshot
    topology: Optional[Dict[str, Any]] = None

    # Spatial scene graph relationships
    relationships: List[SceneRelationship] = field(default_factory=list)

    # Evidence & Explicit Unknowns
    scene_confidence: float = 1.0
    unknowns: List[str] = field(default_factory=list)

    def get_entity_by_id(self, entity_id: str) -> Optional[EntityState]:
        for e in self.entities:
            if e.entity_id == entity_id:
                return e
        return None

    def get_entity_by_seat(self, seat_label: str) -> Optional[EntityState]:
        for e in self.entities:
            if e.seat_label == seat_label:
                return e
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "frame_idx": self.frame_idx,
            "people_count": self.people_count,
            "occupied_seats": self.occupied_seats,
            "total_seats": self.total_seats,
            "average_attention": self.average_attention,
            "scene_confidence": self.scene_confidence,
            "unknowns": list(self.unknowns),
            "topology": self.topology,
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
        }
