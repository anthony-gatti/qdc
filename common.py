"""
QPQ evaluation framework constants and shared utilities.
"""

import math

#  Sequence time units (picoseconds)
SECOND = int(1e12)
MILLISECOND = int(1e9)
MICROSECOND = int(1e6)

# QPQ protocol parameters 

def pairs_per_round(database_size_log: int) -> int:
    """Bell pairs needed per QPQ round.

    Each round: Alice sends log(N) qubit register, receives log(N)+1 register.
    Every qubit needs one Bell pair => 2*log(N) + 1 pairs per round.
    """
    n = database_size_log
    return 2 * n + 1


def pairs_per_query(database_size_log: int) -> int:
    """Bell pairs needed per complete QPQ query (2 rounds)."""
    return 2 * pairs_per_round(database_size_log)


# Reference values:
# N=2^10  (~1K):   n=10, pairs/round=21, pairs/query=42
# N=2^16  (~65K):  n=16, pairs/round=33, pairs/query=66
# N=2^20  (~1M):   n=20, pairs/round=41, pairs/query=82
# N=2^30  (~1B):   n=30, pairs/round=61, pairs/query=122

QPQ_ROUNDS_PER_QUERY = 2