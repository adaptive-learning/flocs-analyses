from collections import Counter
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster.bicluster import SpectralCoclustering
from similarity import edit_distance, visited_sets_distance, compute_visited_set
from robotanik_read import load_problem


# settings
GRID_SIZE = (12, 16)
sns.set()


def test():
    #run_doctests()
    #affinity_propagation_test()
    #spectral_coclustering_test()
    test_cluster_visualization()


def run_doctests():
    import doctest
    doctest.testmod()


def affinity_propagation_test():
    problem = load_problem(problem_id='639')
    programs = problem['attempts']
    n = len(programs)
    print('Number of programs:', n)
    similarity_matrix = get_similarity_matrix(problem, programs)
    af = AffinityPropagation(preference=-2, affinity='precomputed')
    af.fit(similarity_matrix)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    print('Number of clusters:', len(cluster_centers_indices))
    print(cluster_centers_indices)
    print(labels)


def spectral_coclustering_test():
    problem = load_problem(problem_id='639')
    programs = problem['attempts'][:100]
    #similarities = get_similarity_matrix(programs, distance_fn=partial(visited_sets_distance, problem))
    similarities = get_similarity_matrix(programs, distance_fn=edit_distance)
    #plot_similarity_matrix(similarity_matrix)
    model = SpectralCoclustering(n_clusters=10, random_state=0)
    model.fit(similarities)
    #print(model.rows_)
    order = np.argsort(model.row_labels_)
    reordered_similarities = similarities[:, order][order]
    plot_similarity_matrix(reordered_similarities)


def plot_similarity_matrix(similarities):
    #plt.title("Similarity matrix")
    plt.matshow(similarities, cmap=plt.cm.Blues)
    plt.show()


def get_similarity_matrix(programs, distance_fn=edit_distance):
    similarity_matrix = np.array([[-distance_fn(programA, programB)
                                for programA in programs]
                                for programB in programs])
    return similarity_matrix


def test_cluster_visualization():
    problem = load_problem(problem_id='639')
    programs = problem['attempts'][:100]
    similarities = get_similarity_matrix(programs, distance_fn=edit_distance)
    model = SpectralCoclustering(n_clusters=10, random_state=0)
    model.fit(similarities)
    cluster_labels = model.column_labels_
    cluster = pd.Series(programs)[cluster_labels == 0]
    print('Cluster size:', len(cluster))
    visualize_cluster(problem, cluster)



def visualize_cluster(problem, cluster):
    visits = get_mean_visited_set(problem, cluster)
    plt.matshow(visits)
    plt.show()


def get_mean_visited_set(problem, cluster):
    cluster = pd.Series(cluster)
    visited_sets = cluster.apply(partial(compute_visited_set, problem))
    visited_counts = Counter(pos for vs in visited_sets for pos in vs)
    frequencies = np.zeros(GRID_SIZE)
    n = len(cluster)
    for pos, count in visited_counts.items():
        frequencies[pos] = count / n
    return frequencies





if __name__ == "__main__":
    test()
