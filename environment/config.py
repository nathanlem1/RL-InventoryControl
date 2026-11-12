ENV_CONFIG = {
    "episode_length": 5000,
    "num_episodes":200,
    "initial_inventory": 50,
    "max_order": 60,
    "lead_time": 10,
    "action_bins": [0,5,10,15,20,25,30,40],
    "inv_position_scale": 200,
    "demand_history_len": 5,
}

PERSISTENCE = {
    "low" : {
        "transition_matrix": [[0.8,0.2],[0.2,0.8]]
       },
    "medium" : {
        "transition_matrix": [[0.97,0.03],[0.03,0.97]]
    },
    "high" : {
        "transition_matrix": [[0.97,0.03],[0.03,0.97]]
    },
    "three Regime" : {
        "transition_matrix": [[0.90, 0.08, 0.02],[0.07, 0.90, 0.03],[0.1,0.1,0.8]]

    }
}

COSTS = {
    "baseline": {
        "holding_cost": 0.30,
        "stockout_cost": 1.0,
        "fixed_order_cost": 0.2,
        "unit_order_cost": 0.40,
        "selling_price": 2.00
    },
}
