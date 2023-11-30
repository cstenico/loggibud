"""
This baseline is a simple partioning followed by a routing problem.

It uses pure K-Means to partition the problem into K convex regions and them uses the ORTools solver
to solve each subinstance. It's similar to the method proposed by Ruhan et al. [1], but without the
balancing component.

Refs:

[1] R. He, W. Xu, J. Sun and B. Zu, "Balanced K-Means Algorithm for Partitioning Areas in Large-Scale
Vehicle Routing Problem," 2009 Third International Symposium on Intelligent Information Technology
Application, Shanghai, 2009, pp. 87-90, doi: 10.1109/IITA.2009.307. Available at
https://ieeexplore.ieee.org/abstract/document/5369502.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path
import json

import numpy as np
from sklearn.cluster import KMeans

from loggibud.v2.types import VRPPDInstance, VRPPDSolution
from loggibud.v2.baselines.shared.ortools import (
    solve as ortools_solve,
    ORToolsParams,
)
from multiprocessing import Pool

logger = logging.getLogger(__name__)

def ortools_solve_wrapper(subinstance, ortools_params):
    return ortools_solve(subinstance, ortools_params)


@dataclass
class KmeansPartitionORToolsParams:
    fixed_num_clusters: Optional[int] = None
    variable_num_clusters: Optional[int] = None
    seed: int = 0

    ortools_params: Optional[ORToolsParams] = None
    save_to: Optional[str] = None
    multiprocessing: bool = False

    @classmethod
    def get_baseline(cls):
        return cls(
            variable_num_clusters=500,
            ortools_params=ORToolsParams(
                time_limit_ms=120_000,
                solution_limit=1_000,
            ),
        )

def solve(
    instance: VRPPDInstance,
    params: Optional[KmeansPartitionORToolsParams] = None,
) -> Optional[VRPPDSolution]:

    params = params or KmeansPartitionORToolsParams.get_baseline()

    num_demands = len(instance.demands)
    num_clusters = int(
        params.fixed_num_clusters
        or np.ceil(
            num_demands / (params.variable_num_clusters or num_demands)
        )
    )

    logger.info(f"Clustering instance into {num_clusters} subinstances")
    clustering = KMeans(num_clusters, random_state=params.seed)

    points = np.array(
        [[d.point.lng, d.point.lat] for d in instance.demands]
    )
    clusters = clustering.fit_predict(points)

    demands_array = np.array(instance.demands)

    subsinstance_demands = [
        demands_array[clusters == i] for i in range(num_clusters)
    ]

    subinstances = [
        VRPPDInstance(
            name=instance.name,
            demands=subinstance.tolist(),
            depot=instance.depot,
            vehicle_capacity=instance.vehicle_capacity,
            region=instance.region
        )
        for subinstance in subsinstance_demands
    ]

    if params.multiprocessing:
        with Pool(len(subinstances)) as pool:
            args = [(subinstance, params.ortools_params) for subinstance in subinstances]
            subsolutions = pool.starmap(ortools_solve_wrapper, args)
    
    else:
        subsolutions = [
            ortools_solve(subinstance, params.ortools_params)
            for subinstance in subinstances
        ]

    solution = VRPPDSolution(
        name=instance.name,
        vehicles=[v for sol in subsolutions for v in sol.vehicles],
    )

    print(len(solution.vehicles))

    if params.save_to is not None:
        logger.info(f"Saving solution to {params.save_to}")

        dir_path = Path(f"{params.save_to}/{instance.region}/{instance.name}")
        dir_path.mkdir(parents=True, exist_ok=True)

        path = Path(dir_path / f"{instance.name}.json")
        with path.open("w") as file:
            json.dump(asdict(solution), file)
    
    return solution
