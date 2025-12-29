from __future__ import annotations
from typing import List, Tuple, Optional
import math


def compute_budget_efficient(total_budget: float, citizen_votes: List[List[float]]) -> List[float]:
    """
    Q5(b) — Efficient solution for the Generalized Median Algorithm with linear rising phantom functions,
    WITHOUT using binary search.

    Same model as in part (a):
      - n citizens, m subjects.
      - For each subject j, we compute the median of:
            { citizen_votes[i][j] for i=1..n }  union  { f_k(t) for k=1..n-1 }
        where f_k(t) = C * min(1, k*t).

    Goal:
      Find t* in [0,1] such that:
          S(t*) = sum_j median_j(t*) = C
      Then return the vector (median_1(t*), ..., median_m(t*)).

    Key idea:
      S(t) is piecewise-linear in t.
      Breakpoints happen only when:
        (1) A phantom saturates: t = 1/k
        (2) A phantom crosses a citizen vote v: t = v / (C*k)
      On each interval between consecutive breakpoints, S(t) = A + B*t,
      so we can solve t* exactly on that interval.

    Complexity (typical for the assignment sizes):
      - Build breakpoints: O(m * n * (n-1))
      - For each interval, compute medians by merging sorted lists in O(m*n)
      - Exact (finite) termination, no iterative binary search.
    """

    # Input check
    if total_budget < 0:
        raise ValueError("total_budget must be non-negative.")
    if not citizen_votes or not citizen_votes[0]:
        return []

    # Dimensions
    n = len(citizen_votes)
    m = len(citizen_votes[0])
    for row in citizen_votes:
        if len(row) != m:
            raise ValueError("All citizens must vote on the same number of subjects.")
        # all citizens must vote 100% on one subject, and not split their vote between subjects
                

    # Edge case: n=1 -> no phantoms. The algorithm is just the (single) citizen vector.
    # There is no t parameter that can change anything.
    if n == 1:
        return [float(x) for x in citizen_votes[0]]

    # Step 1: Pre-sort citizen votes
    # Pre-sort citizen votes per subject for fast median selection by merging.
    citizens_sorted: List[List[float]] = []
    for j in range(m):
        col = sorted(float(citizen_votes[i][j]) for i in range(n))
        citizens_sorted.append(col)

    # Step 2: Phantom function
    # f_k(t) = C * min(1, k*t), for k = 1..n-1
    def phantom_value(k: int, t: float) -> float:
        return total_budget * min(1.0, k * t)

    # Step 3: Median selection from merged sequences
    # We need the median of 2n-1 values = the n-th smallest (1-indexed).
    # We compute it by merging two sorted sequences:
    #   - citizens_sorted[j] (length n, constant)
    #   - phantoms at t (length n-1, sorted by k because f_k(t) is nondecreasing in k)
    #
    # We also return which source gave the median:
    #   - phantom_k = None if citizen median
    #   - phantom_k = k if phantom median
    #   - saturated = True iff phantom_k*t >= 1 (so that phantom is constant C on this interval)
    # Note not sure if need to differentiate between citizen vs phantom median 
    def nth_of_merged(cit: List[float], t: float) -> Tuple[float, Optional[int], bool]:
        i = 0            # citizen pointer
        k = 1            # phantom index (k=1..n-1)
        taken = 0        # how many items we've taken from the merged order

        while True:
            cv = cit[i] if i < n else math.inf # get citizen value
            pv = phantom_value(k, t) if k <= n - 1 else math.inf # get phantom value

            # Tie-breaking doesn't matter for correctness except at breakpoints.
            # We choose citizen first on ties.
            if cv <= pv:
                taken += 1
                if taken == n: # found the median
                    return cv, None, False # citizen median
                i += 1
            else:
                taken += 1
                if taken == n:
                    return pv, k, (k * t >= 1.0) # phantom median. returns value, element k, saturated. saturated means phantom is constant C on this interval
                k += 1

    # Step 4: the total budget allocated for a given t (S(t))
    def S_at(t: float) -> float:
        return sum(nth_of_merged(citizens_sorted[j], t)[0] for j in range(m))

    # Sanity check: ensure there is a solution t in [0,1].
    # If it doesn't, the input is likely outside the intended model.
    s0, s1 = S_at(0.0), S_at(1.0)
    if s0 - total_budget > 1e-8 or total_budget - s1 > 1e-8: # if even at t=0 you already allocate more than C
        raise ValueError(
            f"No solution t in [0,1] such that S(t)=C. Got S(0)={s0}, S(1)={s1}, C={total_budget}."
        )

    # Step 5: Build the “event points” (breakpoints)
    # Breakpoints where ordering can change:
    #  (1) saturation points t = 1/k
    #  (2) crossing points t = v / (C*k) when v is a citizen vote
    bps = {0.0, 1.0}
    for k in range(1, n):
        bps.add(1.0 / k)

    if total_budget > 0:
        for j in range(m):
            for v in citizens_sorted[j]: # for each citizen vote on subject j
                for k in range(1, n): # for each phantom k=1..n-1
                    t = v / (total_budget * k) # crossing point
                    if 0.0 <= t <= 1.0:
                        bps.add(t) # add crossing point

    # Sort and compress close duplicates (floating-point tolerance).
    sorted_bps = sorted(bps)
    uniq: List[float] = []
    eps_merge = 1e-12
    for x in sorted_bps:
        if not uniq or abs(x - uniq[-1]) > eps_merge:
            uniq.append(x)
    sorted_bps = uniq

    # If we hit *exact* equality at some breakpoint (rare but possible), return immediately.
    for t in sorted_bps:
        if abs(S_at(t) - total_budget) < 1e-9:
            return [float(nth_of_merged(citizens_sorted[j], t)[0]) for j in range(m)]

    # Step 6: Scan each interval and solve exactly
    for left, right in zip(sorted_bps[:-1], sorted_bps[1:]):
        if right - left < 1e-15:
            continue

        t_mid = (left + right) / 2.0  # any interior point identifies the median source on this interval

        # Build S(t)=A + B*t on this interval.
        A = 0.0
        B = 0.0

        for j in range(m):
            _, phantom_k, saturated = nth_of_merged(citizens_sorted[j], t_mid)

            if phantom_k is None:
                # Citizen median -> constant on the interval
                v, _, _ = nth_of_merged(citizens_sorted[j], t_mid)
                A += v
            else:
                if saturated:
                    # Phantom median, saturated -> constant C
                    A += total_budget
                else:
                    # Phantom median, unsaturated -> C*k*t
                    B += total_budget * phantom_k

        # If B=0 then S(t) is constant on this whole interval.
        if abs(B) < 1e-14:
            if abs(A - total_budget) < 1e-9:
                t_star = left
                return [float(nth_of_merged(citizens_sorted[j], t_star)[0]) for j in range(m)]
            continue

        t_star = (total_budget - A) / B
        if t_star < left - 1e-12 or t_star > right + 1e-12:
            continue

        # Clamp for safety (floating point).
        t_star = min(max(t_star, left), right)

        # Final budgets at t*
        budget = [float(nth_of_merged(citizens_sorted[j], t_star)[0]) for j in range(m)]
        return budget

    # Should not reach here for standard inputs; fallback.
    return [float(nth_of_merged(citizens_sorted[j], 1.0)[0]) for j in range(m)]


# --- Examples / run tests (like required) ---
if __name__ == "__main__":
    print("Example 1:")
    print(compute_budget_efficient(100, [[100, 0, 0], [0, 0, 100]]))  # expected [50, 0, 50]

    # not good: must be 100% on one subject לא טובבבבב
    print("\nExample 2:")
    print(compute_budget_efficient(30, [
        [6, 6, 6, 6, 0, 0, 6, 0, 0],
        [0, 0, 6, 6, 6, 6, 0, 6, 0],
        [6, 6, 0, 0, 6, 6, 0, 0, 6],
    ]))

    print("\nExample 3:")
    print(compute_budget_efficient(30, [
        [30, 0, 0],
        [0, 30, 0],
        [0, 0, 30]
    ]))

    print("\nExample 4:")
    print(compute_budget_efficient(100, [
        [100, 0],
        [100, 0],
        [0, 100]
    ]))