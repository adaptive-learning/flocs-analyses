from collections import Counter
import numpy as np
import pylab as plt
from robotanik_read import load_problem
from robotanik_properties import simulate, get_board
from robotanik_read import parse_roboprogram


def test():
    run_doctests()
    #problem = load_problem(problem_id='639')
    #programA = problem['attempts'][0]
    #programB = problem['attempts'][1]
    #d = visited_sets_distance(problem, programA, programB)
    #print(d)


def run_doctests():
    import doctest
    doctest.testmod()


def edit_distance(programA, programB):
    """Cost of edits to transform one program into another
    """
    pass


def world_effect_distance(problem, programA, programB):
    """Distance between observable effects of two programs

    For Robotanik, there are 3 effects:
    1. eaten flowers
    2. recolored fields
    3. final Robotanik position and direction
    """
    pass  # TODO


def eaten_flowers_distance(problem, programA, programB):
    """Distance between the sets of eaten flowers
    # TODO: try replace tuples by parse_roboprogram

    >>> problem = load_problem(problem_id='659')  # Tree of flowers
    >>> programA = parse_roboprogram('_F_F_F_F')
    >>> programB = parse_roboprogram('_F')
    >>> eaten_flowers_distance(problem, programA, programB)
    0.75
    """
    eaten_flowers_A = compute_eaten_flowers(problem, programA)
    eaten_flowers_B = compute_eaten_flowers(problem, programB)
    distance = jaccard_distance(eaten_flowers_A, eaten_flowers_B)
    return distance


# TODO: move to robotanik_properties
def compute_eaten_flowers(problem, program):
    """Compute set of eaten flowers (each flower is encoded by its position)

    >>> problem = load_problem(problem_id='698')  # Crest
    >>> programA = parse_roboprogram('')
    >>> compute_eaten_flowers(problem, programA)
    set()
    >>> programB = parse_roboprogram('rLrL_FbL_1')
    >>> sorted(compute_eaten_flowers(problem, programB))
    [(2, 15), (3, 11), (4, 7), (5, 3)]
    """
    visited = compute_visited_set(problem, program)
    flowers = get_flowers(problem)
    visited_flowers = flowers & visited
    return visited_flowers


# TODO: move to robotanik_properties (?)
def get_flowers(problem):
    """Find a set of flowers in the problem (each flower is encoded by its position)
    >>> problem = load_problem(problem_id='698')
    >>> sorted(get_flowers(problem))
    [(2, 15), (3, 11), (4, 7), (5, 3)]
    """
    board = get_board(problem)
    flowers = set((i,j)
                  for i, row in enumerate(board)
                  for j, field in enumerate(row)
                  if is_flower(field))
    return flowers


# TODO: move to robotanik properties (?)
def is_flower(field):
    return field.isupper()


def visited_sets_distance(problem, programA, programB):
    """Jaccard distance between sets of visted fields

    >>> problem = load_problem(problem_id='639')
    >>> programA = [[('_', 'F'), ('_', 'F'), ('_', 'F')], [], [], [], [], []]
    >>> programB = [[('_', 'L'), ('_', 'F')], [], [], [], []]
    >>> visited_sets_distance(problem, programA, programB)
    0.8
    """
    visited_set_A = compute_visited_set(problem, programA)
    visited_set_B = compute_visited_set(problem, programB)
    distance = jaccard_distance(visited_set_A, visited_set_B)
    return distance


def visited_multisets_distance(problem, programA, programB):
    """Jaccard distance between multisets of visted fields

    >>> problem = load_problem(problem_id='639')
    >>> programA = [[('_', 'F'), ('_', 'F')], [], [], [], [], []]
    >>> programB = [[('_', 'L'), ('_', 'F'), ('_', 'L'), ('_', 'L'), ('_', 'F')], [], [], [], []]
    >>> visited_multisets_distance(problem, programA, programB)
    0.8
    """
    visited_multiset_A = compute_visited_multiset(problem, programA)
    visited_multiset_B = compute_visited_multiset(problem, programB)
    distance = jaccard_multisets_distance(visited_multiset_A, visited_multiset_B)
    return distance


def jaccard_distance(setA, setB):
    """Distance between two sets

    >>> jaccard_distance({1, 2}, {1, 2})
    0.0
    >>> jaccard_distance(set(), set())
    0.0
    >>> jaccard_distance({1, 2}, {3, 4})
    1.0
    >>> jaccard_distance({1, 2, 3}, {1, 2, 4, 5})
    0.6
    """
    intersection_size = len(setA & setB)
    union_size = len(setA | setB)
    distance = (1 - intersection_size / union_size) if union_size > 0 else 0.
    return distance


def jaccard_multisets_distance(multisetA, multisetB):
    """Distance between two multisets (counters)

    >>> jaccard_multisets_distance(Counter('aaab'), Counter('aac'))
    0.6
    """
    intersection_size = sum((multisetA & multisetB).values())
    union_size = sum((multisetA | multisetB).values())
    distance = (1 - intersection_size / union_size) if union_size > 0 else 0.
    return distance


def visited_fields_analysis():
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
    test()
