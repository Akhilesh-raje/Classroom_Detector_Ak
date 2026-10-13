"""
SmartClass AI — Spatial Intelligence Domain Package
Handles physical classroom topologies, grid normalization, seat association, anchor stabilization, and coordinate transforms.
"""
from core.spatial.topology import (
    ClassroomTopology,
    TopologyConfig,
    TopologyDiscovery,
    PhysicalSeat,
    PersonTrack,
)
from core.spatial.seat_associator import (
    SeatAssociator,
    SeatAssignment,
    RowClassifier,
)
from core.spatial.stabilizer import SeatStabilizer
from core.spatial.seat_naming import (
    seat_label,
    parse_seat_label,
    all_seats,
    seats_per_row_labels,
)
from core.spatial.classroom_mapper import ClassroomMapper
from core.spatial.spatial_engine import SpatialEngine
from core.spatial.spatial_engine_v3 import (
    build_topology,
    topology_to_dict,
    render_topology,
)

__all__ = [
    "ClassroomTopology",
    "TopologyConfig",
    "TopologyDiscovery",
    "PhysicalSeat",
    "PersonTrack",
    "SeatAssociator",
    "SeatAssignment",
    "RowClassifier",
    "SeatStabilizer",
    "seat_label",
    "parse_seat_label",
    "all_seats",
    "seats_per_row_labels",
    "ClassroomMapper",
    "SpatialEngine",
    "build_topology",
    "topology_to_dict",
    "render_topology",
]
