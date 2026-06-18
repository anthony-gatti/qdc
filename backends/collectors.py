from sequence.constants import MILLISECOND

from qpq_app import QPQApp
from results import BackendResult, RequestResult


def collect_qpq_results(name_to_app: dict, config: dict, backend_name: str) -> BackendResult:
    """Extract QPQ query results from QPQ apps.

    Pair timestamps are reported in ms relative to each query's round-1 start.
    """
    request_results = []

    for app in name_to_app.values():
        if not isinstance(app, QPQApp):
            continue

        for qpq_result in app.get_results():
            query = app.queries.get(qpq_result.query_id)

            pair_arrivals_ms = []
            round1_completion_ms = None
            round2_completion_ms = None
            pairs_rejected_fidelity = 0
            if query is not None:
                round1_start_ps = 0
                if 1 in query.rounds:
                    round1_start_ps = query.rounds[1].start_time_ps

                for round_num in (1, 2):
                    if round_num not in query.rounds:
                        continue
                    rnd = query.rounds[round_num]
                    if rnd.reservation is None:
                        continue

                    ts_list = app.entanglement_timestamps.get(rnd.reservation, [])
                    for ts_ps in ts_list:
                        pair_arrivals_ms.append((ts_ps - round1_start_ps) / MILLISECOND)
                    pairs_rejected_fidelity += app.low_fidelity_rejects.get(
                        rnd.reservation, 0
                    )

                    if rnd.end_time_ps > 0:
                        completion_ms = (rnd.end_time_ps - round1_start_ps) / MILLISECOND
                        if round_num == 1:
                            round1_completion_ms = completion_ms
                        else:
                            round2_completion_ms = completion_ms

            avg_fid = None
            if qpq_result.success:
                avg_fid = (
                    qpq_result.round1_avg_fidelity
                    + qpq_result.round2_avg_fidelity
                ) / 2

            request_results.append(RequestResult(
                request_id=qpq_result.query_id,
                src=qpq_result.src,
                dst=qpq_result.dst,
                start_time_ps=0,
                time_to_serve_ms=qpq_result.total_time_ms,
                fidelity=avg_fid,
                success=qpq_result.success,
                failure_reason=qpq_result.failure_reason or "",
                pair_arrival_ms=pair_arrivals_ms,
                first_pair_arrival_ms=min(pair_arrivals_ms) if pair_arrivals_ms else None,
                round1_completion_ms=round1_completion_ms,
                round2_completion_ms=round2_completion_ms,
                round1_pairs=qpq_result.round1_pairs,
                round2_pairs=qpq_result.round2_pairs,
                expected_pairs=2 * qpq_result.pairs_per_round,
                pairs_rejected_fidelity=pairs_rejected_fidelity,
            ))

    seed = config.get("topology", {}).get("random_seed", 0)

    return BackendResult(
        backend_name=backend_name,
        seed=seed,
        num_nodes=len(name_to_app),
        request_results=request_results,
    )
