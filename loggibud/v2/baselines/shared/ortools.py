import logging

from typing import Optional
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from loggibud.v2.types import JSONDataclassMixin, VRPPDInstance, VRPPDSolution, VRPPDSolutionVehicle
from loggibud.v2.distances import calculate_distance_matrix_great_circle_m, calculate_distance_matrix_m, OSRMConfig


logger = logging.getLogger(__name__)


@dataclass
class ORToolsParams(JSONDataclassMixin):
    first_solution_strategy: int = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    local_search_metaheuristic: int = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    max_vehicles: Optional[int] = None
    solution_limit: Optional[int] = None
    time_limit_ms: Optional[int] = 60_000

    osrm_config: Optional[OSRMConfig] = None
    """Config for calling OSRM distance service."""

def solve(
    instance: VRPPDInstance,
    params: Optional[ORToolsParams] = None,
) -> Optional[VRPPDSolution]:
    """Solves a VRPPD instance using ORTools"""

    # Initialize parameters if not provided.
    params = params or ORToolsParams()

    # Number of points is the number of deliveries + the origin.
    num_points = len(instance.demands) + 1

    logger.info(f"Solving CVRP instance of size {num_points}.")

    # There's no limit of vehicles, or max(vehicles) = len(deliveries).
    num_vehicles = len(instance.demands)

    manager = pywrapcp.RoutingIndexManager(
        num_points,
        num_vehicles,
        0,  # (Number of nodes, Number of vehicles, Origin index).
    )
    model = pywrapcp.RoutingModel(manager)

    # Unwrap the size index for every point.
    sizes = np.array(
        [0] + [-d.size if d.type == 'DELIVERY' else d.size for d in instance.demands], dtype=np.int32
    )

    def capacity_callback(src):
        src = manager.IndexToNode(src)
        return sizes[src]

    capacity_callback_index = model.RegisterUnaryTransitCallback(
        capacity_callback
    )
    model.AddDimension(
        capacity_callback_index, 
        0, instance.vehicle_capacity,
        False, "Capacity"
    )

    # Unwrap the location/point for every point.
    locations = [instance.depot] + [d.point for d in instance.demands]

    # Compute the distance matrix between points.
    logger.info("Computing distance matrix.")
    if params.osrm_config is not None:
        distance_matrix = (
            calculate_distance_matrix_m(locations, params.osrm_config)
        ).astype(np.int32)
    else:
        distance_matrix = (
            calculate_distance_matrix_great_circle_m(locations)
        ).astype(np.int32)

    def distance_callback(src, dst):
        x = manager.IndexToNode(src)
        y = manager.IndexToNode(dst)
        return distance_matrix[x, y]

    distance_callback_index = model.RegisterTransitCallback(
        distance_callback
    )
    model.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)

    dimension_name = "Distance"
    model.AddDimension(
        distance_callback_index,
        0,  # no slack
        240000,  # vehicle maximum travel distance (240 kilometers)
        True,  # start cumul to zero
        dimension_name,
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = params.first_solution_strategy

    search_parameters.local_search_metaheuristic = (
        params.local_search_metaheuristic
    )

    if params.solution_limit:
        search_parameters.solution_limit = params.solution_limit

    search_parameters.time_limit.FromTimedelta(
        timedelta(microseconds=1e3 * params.time_limit_ms)
    )

    logger.info("Solving CVRP with ORTools.")
    assignment = model.SolveWithParameters(search_parameters)

    # Checking if the feasible solution was found.
    # For more information about the type error:
    # https://developers.google.com/optimization/routing/routing_options
    if assignment:    
        def extract_solution(vehicle_id):
            # Get the start node for route.
            index = model.Start(vehicle_id)

            # Iterate while we don't reach an end node.
            while not model.IsEnd(assignment.Value(model.NextVar(index))):
                next_index = assignment.Value(model.NextVar(index))
                node = manager.IndexToNode(next_index)

                yield instance.demands[node - 1]
                index = next_index

        routes = [
            VRPPDSolutionVehicle(
                origin=instance.depot,
                demands=list(extract_solution(i)),
            )
            for i in range(num_vehicles)
        ]

        # Return only routes that actually leave the depot.
        return VRPPDSolution(
            name=instance.name,
            vehicles=[v for v in routes if len(v.demands)],
        )
    return None