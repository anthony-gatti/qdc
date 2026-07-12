"""
Generates QPQ-style request queues for the evaluation framework.

Produces request tuples in the format expected by ACP's RequestAppTimeToServe:
  (id, src_name, dst_name, start_time, end_time, memo_size, fidelity, entanglement_number)
"""

import numpy as np
from typing import List, Tuple
from common import SECOND, MILLISECOND, pairs_per_round

# Type alias for request tuples
Request = Tuple[int, str, str, int, int, int, float, int]


def generate_qpq_requests(
    router_names: List[str],
    qdc_name: str,
    num_clients: int,
    queries_per_client: int,
    database_size_log: int = 20,
    fidelity_threshold: float = 0.7,
    request_period_s: float = 1.0,
    start_offset_s: float = 2.0,
    request_duration_s: float = 5.0,
    seed: int = 42,
) -> List[Request]:
    """Generate a queue of QPQ entanglement requests.

    For now, each QPQ query is modeled as a SINGLE entanglement request
    (not multi-round). Multi-round semantics will be added in Phase 3
    via a custom QPQ application class.

    The request asks for `pairs_per_round(database_size_log)` Bell pairs
    between a client router and the QDC server.

    Args:
        router_names: list of all router names in the topology.
        qdc_name: name of the QDC server router.
        num_clients: how many client routers generate requests.
        queries_per_client: number of queries each client issues.
        database_size_log: n = log2(N), determines Bell pairs per round.
        fidelity_threshold: minimum acceptable fidelity for the request.
        request_period_s: time between consecutive requests (seconds).
        start_offset_s: delay before first request (seconds, allows ACP warmup).
        request_duration_s: time window for each request to complete.
        seed: random seed for client selection.

    Returns:
        List of request tuples sorted by start_time, ready to feed
        to RequestAppTimeToServe.start().
    """
    rng = np.random.default_rng(seed)

    # Select client nodes (exclude QDC itself)
    available_clients = [r for r in router_names if r != qdc_name]
    if num_clients > len(available_clients):
        num_clients = len(available_clients)

    clients = list(rng.choice(available_clients, size=num_clients, replace=False))

    # Bell pairs needed per request
    memo_size = 1  # SeQUeNCe requests 1 memory pair at a time;
    # the "pairs_per_round" scaling is a workload parameter
    # that affects how many sequential requests a query generates.

    entanglement_number = 1

    requests = []
    req_id = 0

    for client in clients:
        for q in range(queries_per_client):
            start_time = int((start_offset_s + q * request_period_s) * SECOND)
            end_time = start_time + int(request_duration_s * SECOND)

            requests.append((
                req_id,
                client,       # src (initiator)
                qdc_name,     # dst (responder = QDC server)
                start_time,
                end_time,
                memo_size,
                fidelity_threshold,
                entanglement_number,
            ))
            req_id += 1

    # Sort by start time
    requests.sort(key=lambda r: r[3])
    return requests


def generate_qpq_queries(
    router_names: List[str],
    qdc_name: str,
    num_clients: int,
    queries_per_client: int,
    database_size_log: int = 20,
    fidelity_threshold: float = 0.7,
    round_deadline_s: float = 5.0,
    request_period_s: float = 1.0,
    start_offset_s: float = 2.0,
    reservation_duration_s: float = 15.0,
    seed: int = 42,
) -> List[dict]:
    """Generate QPQ query specs for the QPQ application.

    Each query spec is a dict consumed by QPQApp.submit_query().

    Args:
        router_names: list of all router names in the topology.
        qdc_name: name of the QDC server router.
        num_clients: how many client routers generate queries.
        queries_per_client: number of QPQ queries each client issues.
        database_size_log: n = log2(N), determines Bell pairs per round.
        fidelity_threshold: minimum acceptable fidelity per pair.
        round_deadline_s: max seconds to deliver all pairs in one round.
        request_period_s: time between consecutive queries per client.
        start_offset_s: delay before first query (ACP warmup).
        reservation_duration_s: time window for each query to complete
            (both rounds must finish within this window).
        seed: random seed for client selection.

    Returns:
        List of query spec dicts sorted by start_time.
    """
    rng = np.random.default_rng(seed)

    available_clients = [r for r in router_names if r != qdc_name]
    if num_clients > len(available_clients):
        num_clients = len(available_clients)

    clients = list(rng.choice(available_clients, size=num_clients, replace=False))

    queries = []
    query_id = 0

    for client in clients:
        for q in range(queries_per_client):
            start_time = int((start_offset_s + q * request_period_s) * SECOND)
            end_time = start_time + int(reservation_duration_s * SECOND)
            round_deadline_ps = int(round_deadline_s * SECOND)

            queries.append({
                "query_id": query_id,
                "src": client,
                "dst": qdc_name,
                "start_time": start_time,
                "end_time": end_time,
                "database_size_log": database_size_log,
                "fidelity": fidelity_threshold,
                "round_deadline_ps": round_deadline_ps,
            })
            query_id += 1

    queries.sort(key=lambda q: q["start_time"])
    return queries


def generate_uniform_requests(
    router_names: List[str],
    num_requests: int,
    request_period_s: float = 1.0,
    start_offset_s: float = 2.0,
    request_duration_s: float = 5.0,
    fidelity: float = 0.7,
    seed: int = 42,
) -> List[Request]:
    """Generate requests between random router pairs.

    Useful for testing without QPQ-specific structure.
    Matches the format used in ACP's TrafficMatrix.
    """
    rng = np.random.default_rng(seed)
    requests = []

    for i in range(num_requests):
        src, dst = rng.choice(router_names, size=2, replace=False)
        start_time = int((start_offset_s + i * request_period_s) * SECOND)
        end_time = start_time + int(request_duration_s * SECOND)

        requests.append((
            i,
            str(src),
            str(dst),
            start_time,
            end_time,
            1,           # memo_size
            fidelity,
            1,           # entanglement_number
        ))

    requests.sort(key=lambda r: r[3])
    return requests