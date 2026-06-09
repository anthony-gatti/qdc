def get_backend(name: str, config: dict):
    """Construct a backend by name.

    ACP is imported lazily so the core framework can run without ACP installed.
    """
    name = name.lower()

    if name == "odo":
        from backends.odo_backend import ODOBackend
        return ODOBackend()

    if name == "acp":
        try:
            from backends.acp_backend import ACPBackend
        except ImportError as e:
            raise RuntimeError(
                "ACP backend requested, but ACP could not be imported. "
                "Install or patch ACP, then set QDC_ACP_DIR to its directory."
            ) from e

        hw = config.get("hardware", {})
        return ACPBackend(
            adaptive_max_memory=hw.get("acp_memory", 8),
            update_prob=True,
        )

    raise ValueError(f"Unknown backend: {name}")