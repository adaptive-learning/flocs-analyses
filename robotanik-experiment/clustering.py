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
from similarity import program_to_code
from robotanik_read import load_problem


# settings
GRID_SIZE = (12, 16)
sns.set()
sns.set_context("paper")


def test():
    #run_doctests()
    #affinity_propagation_test()
    #spectral_coclustering_test()
    test_clusters_visualization()


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


def test_clusters_visualization():
    problem = load_problem(problem_id='639')
    programs = problem['attempts'][:100]
    n_clusters = 8

    # find clusters
    similarities = get_similarity_matrix(programs, distance_fn=edit_distance)
    model = SpectralCoclustering(n_clusters=n_clusters, random_state=0)
    model.fit(similarities)
    cluster_labels = model.column_labels_
    df = pd.DataFrame.from_dict({'program': programs, 'cluster_id': cluster_labels})

    # compute cluster descriptions
    clusters = df.groupby('cluster_id')
    cluster_descriptions = {cluster_id: get_cluster_description(cluster['program'])
                            for cluster_id, cluster in clusters}
    df['cluster_description'] = df['cluster_id'].map(cluster_descriptions)

    # plot all clusters
    grid = sns.FacetGrid(df, col='cluster_description', col_wrap=4)
        # 'cluster_description' column is used to get meaningful titles
    visualize_cluster_fn = partial(visualize_cluster, problem)
    visualize_cluster_fn.__module__ = __name__  # required by seaborn
    grid = grid.map(visualize_cluster_fn, 'program')
    grid.set_axis_labels('', '')
    grid.set_titles('{col_name}')
    #plt.show()
    plt.savefig('clusters.pdf')


def get_cluster_description(cluster):
    medoid = get_cluster_medoid(cluster, distance_fn=edit_distance)
        # TODO: refactor to use the same distance fn/matrix as used for clustering
    repr_code = program_to_code(medoid)
    avg_edits = int(get_median_distance(medoid, cluster, distance_fn=edit_distance))
    max_edits = int(get_max_distance(medoid, cluster, distance_fn=edit_distance))
    description = '{code} ~ {avg_edits}/{max_edits} edits'.format(
            code=repr_code, avg_edits=avg_edits, max_edits=max_edits)
    return description


def test_single_cluster_visualization():
    problem = load_problem(problem_id='639')
    programs = problem['attempts'][:100]
    similarities = get_similarity_matrix(programs, distance_fn=edit_distance)
    model = SpectralCoclustering(n_clusters=10, random_state=0)
    model.fit(similarities)
    cluster_labels = model.column_labels_
    cluster = pd.Series(programs)[cluster_labels == 0]
    visualize_cluster(problem, cluster)
    plt.show()



def visualize_cluster(problem, cluster, **kwargs):
    visits = get_mean_visited_set(problem, cluster)
    sns.heatmap(visits, cmap='Blues', square=True, linewidths=0.5, linecolor='white', **kwargs)
    #plt.title(description)
    #plt.suptitle(description)


def get_cluster_medoid(cluster, distance_fn):
    programs = np.array(cluster)
    distances = get_distance_matrix(programs, distance_fn)
    medoid_ix = np.argmin(distances.sum(axis=0))
    medoid = programs[medoid_ix]
    return medoid


def get_median_distance(medoid, cluster, distance_fn=edit_distance):
    med_dist = cluster.apply(partial(distance_fn, medoid)).median()
    return med_dist


def get_max_distance(medoid, cluster, distance_fn=edit_distance):
    max_dist = cluster.apply(partial(distance_fn, medoid)).max()
    return max_dist


def get_mean_visited_set(problem, cluster):
    cluster = pd.Series(cluster)
    visited_sets = cluster.apply(partial(compute_visited_set, problem))
    visited_counts = Counter(pos for vs in visited_sets for pos in vs)
    frequencies = np.zeros(GRID_SIZE)
    n = len(cluster)
    for pos, count in visited_counts.items():
        frequencies[pos] = count / n
    return frequencies


def get_similarity_matrix(programs, distance_fn=edit_distance):
    return (-1) * get_distance_matrix(programs, distance_fn)


def get_distance_matrix(programs, distance_fn=edit_distance):
    distance_matrix = np.array([[distance_fn(programA, programB)
                                for programA in programs]
                                for programB in programs])
    return distance_matrix





if __name__ == "__main__":
    test()
