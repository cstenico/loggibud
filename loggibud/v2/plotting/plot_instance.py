import logging
from argparse import ArgumentParser

import numpy as np
import folium

from ..types import VRPPDInstance, MixedProblemInstance, DemandType


logger = logging.getLogger(__name__)


def plot_vrpdp_instance(instance: VRPPDInstance):

    depot = instance.depot
    delivery_points = [demand.point for demand in instance.demands if demand.type == DemandType.DELIVERY]
    pickup_points = [demand.point for demand in instance.demands if demand.type == DemandType.PICKUP]


    # create a map
    m = folium.Map(
        location=(depot.lat, depot.lng),
        zoom_start=12,
        tiles="cartodbpositron",
    )

    for point in delivery_points:
        folium.CircleMarker(
            [point.lat, point.lng], color="blue", radius=1, weight=1
        ).add_to(m)

    for point in pickup_points:
        folium.CircleMarker(
            [point.lat, point.lng], color="green", radius=1, weight=1
        ).add_to(m)

    folium.CircleMarker(
        [depot.lat, depot.lng], color="red", radius=3, weight=5
    ).add_to(m)

    return m


def plot_mixed_instance(instance: MixedProblemInstance):

    points = [demand.point for demand in instance.demands]
    center_lat = np.mean([p.lat for p in points])
    center_lng = np.mean([p.lng for p in points])

    # create a map
    m = folium.Map(
        location=(center_lat, center_lng),
        zoom_start=12,
        tiles="cartodbpositron",
    )

    delivery_points = [demand.point for demand in instance.demands if demand.type == DemandType.DELIVERY]
    pickup_points = [demand.point for demand in instance.demands if demand.type == DemandType.PICKUP]

    for point in delivery_points:
        folium.CircleMarker(
            [point.lat, point.lng], color="blue", radius=1, weight=1
        ).add_to(m)

    for point in pickup_points:
        folium.CircleMarker(
            [point.lat, point.lng], color="green", radius=1, weight=1
        ).add_to(m)

    return m


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--vrpdp", type=str)
    parser.add_argument("--mixed", type=str)

    args = parser.parse_args()

    # Load instance and heuristic params.

    if args.vrpdp:
        instance = VRPPDInstance.from_file(args.vrpdp)
        m = plot_vrpdp_instance(instance)

    elif args.mixed:
        instance = MixedProblemInstance.from_file(args.mixed)
        m = plot_mixed_instance(instance)

    m.save("map.html")
