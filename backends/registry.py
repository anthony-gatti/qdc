def get_backend(name: str, config: dict):
    name = name.lower()
    hw = config.get("hardware", {})

    if name == "odo":
        from backends.odo_backend import ODOBackend
        return ODOBackend()

    if name == "acp":
        from backends.acp_backend import ACPBackend

        return ACPBackend(
            adaptive_max_memory=hw.get("acp_memory", 8),
            update_prob=True,
        )

    if name.startswith("acp_m") and name[5:].isdigit():
        from backends.acp_backend import ACPBackend

        memory_budget = int(name[5:])
        return ACPBackend(
            adaptive_max_memory=memory_budget,
            update_prob=True,
            name_override=f"acp_m{memory_budget}",
        )

    if name == "acp_no_bg":
        from backends.acp_backend import ACPBackend

        return ACPBackend(
            adaptive_max_memory=hw.get("acp_memory", 8),
            update_prob=True,
            background_enabled=False,
            name_override="acp_no_bg",
        )

    raise ValueError(f"Unknown backend: {name}")
