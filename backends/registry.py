def get_backend(name: str, config: dict):
    name = name.lower()
    hw = config.get("hardware", {})
    acp_config = config.get("acp", {})
    default_application_priority = acp_config.get("application_priority", True)

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

    for suffix, application_priority in (("_priority", True), ("_continuous", False)):
        if name.startswith("acp_m") and name.endswith(suffix):
            budget_text = name[5:-len(suffix)]
            if budget_text.isdigit():
                from backends.acp_backend import ACPBackend

                memory_budget = int(budget_text)
                return ACPBackend(
                    adaptive_max_memory=memory_budget,
                    update_prob=True,
                    application_priority=application_priority,
                    name_override=f"acp_m{memory_budget}{suffix}",
                )

    if name == "acp_no_bg":
        from backends.acp_backend import ACPBackend

        return ACPBackend(
            adaptive_max_memory=hw.get("acp_memory", 8),
            update_prob=True,
            background_enabled=False,
            application_priority=False,
            name_override="acp_no_bg",
        )

    raise ValueError(f"Unknown backend: {name}")
