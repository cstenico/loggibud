import logging
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from argparse import ArgumentParser
from typing import Optional

from ..distances import calculate_distance_matrix_great_circle_m, OSRMConfig
from ..types import VRPPDInstance, VRPPDSolution

logger = logging.getLogger(__name__)

@dataclass
class SolutionEvaluation:
    instance_name: str
    delivery_count: int
    pickup_count: int
    route_count: int
    total_distance_km: float
    average_distance_stop: float


def evaluate_solution(
    instance: VRPPDInstance,
    solution: VRPPDSolution,
    config: Optional[OSRMConfig] = None,
    save_to: Optional[str] = None
) -> SolutionEvaluation:

    # Check if all demands are present.
    solution_demands = set(d for v in solution.vehicles for d in v.demands)
    assert solution_demands == set(instance.demands)

    # Check if max capacity is respected.
    max_capacity = max(
        sum(d.size for d in v.demands) for v in solution.vehicles
    )
    assert max_capacity <= instance.vehicle_capacity

    # Check if maximum number of origins is consistent.
    origins = set([v.origin for v in solution.vehicles])
    assert len(origins) <= 1

    route_distances_m = [
        calculate_distance_matrix_great_circle_m(v.circuit, config=config)
        for v in solution.vehicles
    ]

    # Convert to km.
    sum_distance_km = round(sum(route_distances_m) / 1_000, 4)

    # Calculate the average distance per stop in kilometers for each route
    average_distance_stop_per_route_in_km = [
        round(distance / 1000 / len(vehicle.demands), 4)
        for distance, vehicle in zip(route_distances_m, solution.vehicles)
    ]

    # Calculate the overall average distance per stop
    average_distance_stop = round(sum(average_distance_stop_per_route_in_km) / len(solution.vehicles), 4)

    solution_evaluation = SolutionEvaluation(
        instance_name=solution.name,
        delivery_count=len([demand for demand in solution.demands if demand.type == 'DELIVERY']),
        pickup_count=len([demand for demand in solution.demands if demand.type == 'PICKUP']),
        route_count=len(solution.vehicles),
        total_distance_km=sum_distance_km,
        average_distance_stop=average_distance_stop
    )

    if save_to is not None:
        logger.info(f"Saving evaluation to {save_to}")

        dir_path = Path(f"{save_to}")
        dir_path.mkdir(parents=True, exist_ok=True)

        path = Path(dir_path / "kmeans.json")
        with path.open("w") as file:
            json.dump(asdict(solution_evaluation), file)
 
    return solution_evaluation


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--instances", type=str, required=True)
    parser.add_argument("--solutions", type=str, required=True)

    args = parser.parse_args()

    instances_path = Path(args.instances)
    solutions_path = Path(args.solutions)

    if instances_path.is_file() and solutions_path.is_file():
        instances = {"": VRPPDInstance.from_file(instances_path)}
        solutions = {"": VRPPDSolution.from_file(solutions_path)}

    elif instances_path.is_dir() and solutions_path.is_dir():
        instances = {
            f.stem: VRPPDInstance.from_file(f) for f in instances_path.iterdir()
        }
        solutions = {
            f.stem: VRPPDSolution.from_file(f) for f in solutions_path.iterdir()
        }

    else:
        raise ValueError("input files do not match, use files or directories.")

    if set(instances) != set(solutions):
        raise ValueError(
            "input files do not match, the solutions and instances should be the same."
        )

    stems = instances.keys()

    results = [
        evaluate_solution(instances[stem], solutions[stem]) for stem in stems
    ]

    print(sum(results))
