def get_backend(name: str, config: dict):
    name = name.lower()
    hw = config.get("hardware", {})
    algorithm = config.get("algorithm", {})

    if name == "odo":
        from backends.clean_backend import CleanAlgorithmBackend
        return CleanAlgorithmBackend(algorithm_name="odo", name_override="odo")

    if name == "acp":
        from backends.clean_backend import CleanAlgorithmBackend

        return CleanAlgorithmBackend(
            algorithm_name="acp_freshest",
            adaptive_max_memory=algorithm.get(
                "adaptive_max_memory",
                hw.get("acp_memory", 5),
            ),
            name_override="acp",
        )

    if name.startswith("acp_m") and name[5:].isdigit():
        from backends.clean_backend import CleanAlgorithmBackend

        memory_budget = int(name[5:])
        return CleanAlgorithmBackend(
            algorithm_name="acp_freshest",
            adaptive_max_memory=memory_budget,
            name_override=f"acp_m{memory_budget}",
        )

    raise ValueError(f"Unknown backend: {name}")
