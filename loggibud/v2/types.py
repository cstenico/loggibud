import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Union

from dacite import from_dict
from enum import Enum


class JSONDataclassMixin:
    """Mixin for adding JSON file capabilities to Python dataclasses."""

    @classmethod
    def from_file(cls, path: Union[Path, str]) -> "JSONDataclassMixin":
        """Load dataclass instance from provided file path."""

        with open(path) as f:
            data = json.load(f)

        return from_dict(cls, data)

    def to_file(self, path: Union[Path, str]) -> None:
        """Save dataclass instance to provided file path."""

        with open(path, "w") as f:
            json.dump(asdict(self), f)

        return


@dataclass(unsafe_hash=True)
class Point:
    """Point in earth. Assumes a geodesical projection."""

    lng: float
    """Longitude (x axis)."""

    lat: float
    """Latitude (y axis)."""


class DemandType(str, Enum):
    PICKUP = 'PICKUP'
    DELIVERY = 'DELIVERY'


@dataclass(unsafe_hash=True)
class Demand:
    """A delivery or pickup request."""

    id: str
    """Unique id."""

    point: Point
    """Delivery or Pickup location."""

    size: int
    """Size it occupies in the vehicle (considered 1-D for simplicity)."""

    type: str
    """Type of the demand"""


@dataclass
class MixedProblemInstance(JSONDataclassMixin):
    name: str
    """Unique name of this instance."""

    region: str
    """Region name."""

    max_hubs: int
    """Maximum number of hubs allowed in the solution."""

    vehicle_capacity: int
    """Maximum sum of sizes per vehicle allowed in the solution."""

    demands: List[Demand]
    """List of demands to be solved."""


@dataclass
class VRPPDInstance(JSONDataclassMixin):
    name: str
    """Unique name of this instance."""

    region: str
    """Region name."""

    depot: Point
    """Location of the origin hub."""

    vehicle_capacity: int
    """Maximum sum of sizes per vehicle allowed in the solution."""

    demands: List[Demand]
    """List of demands to be solved."""


@dataclass
class VRPPDSolutionVehicle:

    origin: Point
    """Location of the origin hub."""

    demands: List[Demand]
    """Ordered list of demands from the vehicle."""

    @property
    def circuit(self) -> List[Point]:
        return (
            [self.origin] + [d.point for d in self.demands] + [self.origin]
        )

    @property
    def occupation(self) -> int:
        return sum([d.size for d in self.demands])


@dataclass
class VRPPDSolution(JSONDataclassMixin):
    name: str
    vehicles: List[VRPPDSolutionVehicle]

    @property
    def demands(self):
        return [d for v in self.vehicles for d in v.demands]
