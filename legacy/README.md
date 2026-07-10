# Legacy material

`pre_rebuild_acp/` contains the incomplete ACP-on-v1.0.0 port, its associated
QPQ diagnostics, and its historical configuration. It depends on the removed
`external/acp` package and is not imported by any supported command.

`pre_plugin_qpq/` contains the validated ODO-era QPQ application, workload
generator, collector, and demand diagnostics that preceded the common workload
and demand-service interfaces.

It is retained only as historical reference. The supported ACP implementation
is the common SeQUeNCe runtime in `backends/sequence/`, driven by the entrypoints
in `experiments/`.
