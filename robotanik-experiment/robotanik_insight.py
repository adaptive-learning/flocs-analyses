from robotanik_read import *
from robotanik_properties import *
import numpy as np
import pylab as plt
from sklearn.manifold import TSNE

def command_similarity(c1, c2):
    all_commands, same_commands = 0, 0
    assert len(c1) == len(c2)
    for i in range(len(c1)):
        for j in range(max(len(c1[i]), len(c2[i]))):
            all_commands += 2
            if j < len(c1[i]) and j < len(c2[i]):
                if c1[i][j][0] == c2[i][j][0]: same_commands += 1
                if c1[i][j][1] == c2[i][j][1]: same_commands += 1
    #print(same_commands, all_commands)
    return same_commands / all_commands

def jaccard(set1, set2):
    return len(set1.intersection(set2)) / len(set1.union(set2))

def visited_similarity(problem, c1, c2):
    vis1 = simulate(problem, c1)["visitedSet"]
    vis2 = simulate(problem, c2)["visitedSet"]
    return jaccard(vis1, vis2)

def test():
    prob = problems['659']
    #print(prob)
    sol = process_roboprogram("_FbRb1b1bRg1rLrL_F")
    print(sol)
    sol2 = process_roboprogram("_FbLb1g1rLrL_F_Fb1_F")
    #info = simulate(prob, sol2, True)
    #print(info)
    print(command_similarity(sol, sol2))
    print(visited_similarity(prob, sol, sol2))

def analyze_attempts(problem_id, n = 100):
    problem = problems[problem_id]
    attempts = get_attempts(get_path_for_problem(problem_id), n)
    visited_set = [ simulate(problem, c)["visitedSet"] for c in attempts ]
    #print(visited_set)
    n = len(attempts)
    cor = np.zeros((n, n))
    for i in range(n):
        print(i)
        for j in range(n):
            cor[i,j] = (command_similarity(attempts[i], attempts[j]) + jaccard(visited_set[i], visited_set[j])) / 2
            #cor[i,j] = command_similarity(attempts[i], attempts[j])
            #cor[i,j] = jaccard(visited_set[i], visited_set[j])
    #plt.imshow(cor)
    #plt.figure()
    model = TSNE(learning_rate = 150, perplexity=30, n_iter=5000)
    #model = TSNE()
    results = model.fit_transform(cor)
    for i in range(n):
        plt.plot(results[i][0], results[i][1], "o")
    plt.show()

def metric_correlation(problem_id, n = 100):
    problem = problems[problem_id]
    attempts = get_attempts(get_path_for_problem(problem_id), n)
    visited_set = [ simulate(problem, c)["visitedSet"] for c in attempts ]
    for i in range(n):
        for j in range(n):
            plt.plot(command_similarity(attempts[i], attempts[j]), jaccard(visited_set[i], visited_set[j]), "o")
    plt.xlabel("command similarity")
    plt.ylabel("visited similarity")
    plt.show()

def encode_attempt(attempt):
    command_code = { '_': 1, 'r': 2, 'g':3, 'b':4, 'F': 1, 'R': 2, 'L': 3, '1': 4, '2': 5, '3':6, '4':7, '5':8  }
    code = []
    for fun in range(5):
        for i in range(10):
            if fun < len(attempt) and i < len(attempt[fun]):
                code.append(command_code.get(attempt[fun][i][0], 10))
                code.append(command_code.get(attempt[fun][i][1], 10))
            else:
                code.append(0)
                code.append(0)
    return code

def encode_visited(visited):
    board = [ 0 for i in range(12*16) ]
    for x, y in visited:
        board[x*12 + y] = 1
    return board

def properties_projection(problem_id, n = 100):
    problem = problems[problem_id]
    attempts = get_attempts(get_path_for_problem(problem_id), n)
    properties = []
    for i in range(n):
        properties.append(encode_attempt(attempts[i]) + encode_visited(simulate(problem, attempts[i])["visitedSet"]))
        #print(properties[-1])
    model = TSNE(learning_rate = 100, perplexity=40, n_iter=15000)
    #model = TSNE()
    results = model.fit_transform(properties)
    for i in range(n):
        plt.plot(results[i][0], results[i][1], "o")
    plt.show()


def get_path_for_problem(problem_id):
    return "../data/robotanik/task"+str(problem_id)+".txt"


problems = parse_problems()
#analyze_attempts('698', 200)
#analyze_attempts('659', 500)
#analyze_attempts('639', 500)
#metric_correlation('659', 100)
properties_projection('698', 500)
#properties_projection('659', 500)
#properties_projection('639', 500)
