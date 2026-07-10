def get_backend(name: str, config: dict):
    name = name.lower()
    hw = config.get("hardware", {})

    if name == "odo":
        from backends.odo_backend import ODOBackend
        return ODOBackend()

    if name == "acp":
        from backends.clean_backend import CleanAlgorithmBackend

        return CleanAlgorithmBackend(
            algorithm_name="acp_freshest",
            adaptive_max_memory=hw.get("acp_memory", 8),
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
