from collections import Counter
import numpy as np
import pylab as plt
from robotanik_read import load_problem
from robotanik_properties import simulate


def main():
    problem = load_problem(problem_id='639')
    attempts = problem['attempts']
    #print(compute_visited_set(problem, attempts[0]))
    #print(compute_visited_multiset(problem, attempts[0]))
    #simulate(problem, attempts[0], animate=True)

    # histogram for set-world-states
    visited_sets = [frozenset(compute_visited_set(problem, attempt)) for attempt in attempts]
    visited_sets_groups = Counter(visited_sets)
    print(list(reversed(sorted(visited_sets_groups.values()))))

    # histogram for multiset-world-states
    visited_multisets = [compute_visited_multiset(problem, attempt) for attempt in attempts]
    visited_sets_groups = Counter(tuple(sorted(m.items())) for m in visited_multisets)
    frequencies = list(reversed(sorted(visited_sets_groups.values())))
    print(frequencies)

    # if we take 10 most common multiset-world states, how many of the total
    # number of states it is?
    print(sum(frequencies[:10])/len(attempts))



def compute_visited_set(problem, attempt):
    simulation_result = simulate(problem, attempt)
    return simulation_result['visitedSet']


def compute_visited_multiset(problem, attempt):
    simulation_result = simulate(problem, attempt)
    trace = simulation_result['visited']
    counter = Counter(trace)
    return counter


if __name__ == "__main__":
    main()
