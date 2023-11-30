import logging

from .generators import (
    MixedGenerationConfig,
    VRPPDGenerationConfig,
    generate_census_instances,
    generate_cvrp_subinstances,
)

DELIVERY_CONFIGS = {
    "rj": MixedGenerationConfig(
        name="rj",
        num_train_instances=90,
        num_dev_instances=30,
        revenue_income_ratio=1e-4,
        num_deliveries_average=28531,
        num_deliveries_range=4430,
        num_pickups_average=35664,
        num_pickups_range=5538,
        vehicle_capacity=180,
        max_size=10,
        max_hubs=7,
        save_to="./data/125/pickup-and-delivery-instances-1.0",
    ),
    "df": MixedGenerationConfig(
        name="df",
        num_train_instances=90,
        num_dev_instances=30,
        revenue_income_ratio=1e-4,
        num_deliveries_average=9865,
        num_deliveries_range=2161,
        num_pickups_average=12331,
        num_pickups_range=2701,
        vehicle_capacity=180,
        max_size=10,
        max_hubs=3,
        save_to="./data/125/pickup-and-delivery-instances-1.0",
    ),
    "pa": MixedGenerationConfig(
        name="pa",
        num_train_instances=90,
        num_dev_instances=30,
        revenue_income_ratio=1e-4,
        num_deliveries_average=4510,
        num_deliveries_range=956,
        num_pickups_average=5638,
        num_pickups_range=1195,
        vehicle_capacity=180,
        max_size=10,
        max_hubs=2,
        save_to="./data/125/pickup-and-delivery-instances-1.0",
    ),
}


VRPPD_CONFIGS = {
    "rj": VRPPDGenerationConfig(
        name="rj",
        num_hubs=6,
        num_clusters=256,
        vehicle_capacity=180,
        save_to="./data/125/vrppd-instances-1.0",
    ),
    "df": VRPPDGenerationConfig(
        name="df",
        num_hubs=3,
        num_clusters=256,
        vehicle_capacity=180,
        save_to="./data/125/vrppd-instances-1.0",
    ),
    "pa": VRPPDGenerationConfig(
        name="pa",
        num_hubs=2,
        num_clusters=256,
        vehicle_capacity=180,
        save_to="./data/125/vrppd-instances-1.0",
    ),
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    for instance in DELIVERY_CONFIGS:
        config = DELIVERY_CONFIGS[instance]
        delivery_result = generate_census_instances(config)

        cvrp_config = VRPPD_CONFIGS.get(instance)

        if cvrp_config:
            generate_cvrp_subinstances(cvrp_config, delivery_result)
