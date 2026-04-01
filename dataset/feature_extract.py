import tskit
import pyslim
import numpy as np
import pandas as pd
from numba import njit, prange
import sys

TREE_FILE = sys.argv[1]
OUTPUT_TABLE = sys.argv[2]
FREQ_FILE = "/Users/ekaterina/Desktop/RESEARCH/SLiM/freq_output.txt"
DENISOVAN_POP = 1
TIBETAN_POP = 6
N_COLUMNS_TO_SAMPLE = 1000

@njit
def has_ancestry_in_tree(node_id, parent_row, node_population, target_population):
    u = node_id
    while u != -1:
        if node_population[u] == target_population:
            return 1
        u = parent_row[u]
    return 0


@njit(parallel=True)
def build_haplotype_matrices_numba(
    tibetan_node_pairs,
    parent_matrix,
    node_population,
    denisovan_pop,
):
    n_inds = tibetan_node_pairs.shape[0]
    n_intervals = parent_matrix.shape[0]
    hap0_matrix = np.zeros((n_inds, n_intervals), dtype=np.int8)
    hap1_matrix = np.zeros((n_inds, n_intervals), dtype=np.int8)
    for j in prange(n_intervals):
        parent_row = parent_matrix[j]
        for i in range(n_inds):
            h0 = tibetan_node_pairs[i, 0]
            h1 = tibetan_node_pairs[i, 1]
            hap0_matrix[i, j] = has_ancestry_in_tree(h0, parent_row, node_population, denisovan_pop)
            hap1_matrix[i, j] = has_ancestry_in_tree(h1, parent_row, node_population, denisovan_pop)
    return hap0_matrix, hap1_matrix


@njit
def block_length_at_position_numba(row, pos, target_value):
    if row[pos] != target_value:
        return 0
    left = pos
    while left > 0 and row[left - 1] == target_value:
        left -= 1
    right = pos
    n = row.shape[0]
    while right < n - 1 and row[right + 1] == target_value:
        right += 1
    return right - left + 1


@njit(parallel=True)
def compute_column_stats_numba(haplotype_matrix, chosen_cols):
    n_rows, _ = haplotype_matrix.shape
    n_cols = chosen_cols.shape[0]

    freq = np.empty(n_cols, dtype=np.float64)
    mean_nonintro = np.empty(n_cols, dtype=np.float64)
    mean_intro = np.empty(n_cols, dtype=np.float64)
    var_nonintro = np.empty(n_cols, dtype=np.float64)
    var_intro = np.empty(n_cols, dtype=np.float64)

    for k in prange(n_cols):
        col = chosen_cols[k]

        intro_count = 0
        nonintro_count = 0
        intro_sum = 0.0
        nonintro_sum = 0.0
        intro_sq_sum = 0.0
        nonintro_sq_sum = 0.0
        ones_total = 0

        for r in range(n_rows):
            value = haplotype_matrix[r, col]
            ones_total += value
            row = haplotype_matrix[r]

            if value == 1:
                length = block_length_at_position_numba(row, col, 1)
                intro_count += 1
                intro_sum += length
                intro_sq_sum += length * length
            else:
                length = block_length_at_position_numba(row, col, 0)
                nonintro_count += 1
                nonintro_sum += length
                nonintro_sq_sum += length * length

        freq[k] = ones_total / n_rows

        if nonintro_count > 0:
            mn0 = nonintro_sum / nonintro_count
            mean_nonintro[k] = mn0
            var_nonintro[k] = (nonintro_sq_sum / nonintro_count) - (mn0 * mn0)
        else:
            mean_nonintro[k] = 0.0
            var_nonintro[k] = 0.0

        if intro_count > 0:
            mn1 = intro_sum / intro_count
            mean_intro[k] = mn1
            var_intro[k] = (intro_sq_sum / intro_count) - (mn1 * mn1)
        else:
            mean_intro[k] = 0.0
            var_intro[k] = 0.0

    return freq, mean_nonintro, mean_intro, var_nonintro, var_intro


freqs = np.loadtxt(FREQ_FILE)
ts = tskit.load(TREE_FILE)

alive_inds = pyslim.individuals_alive_at(ts, 0)

tibetan_inds = []
for ind_id in alive_inds:
    ind = ts.individual(ind_id)
    nodes = [n for n in ind.nodes if n != tskit.NULL]
    if len(nodes) == 0:
        continue

    node_pops = [ts.node(n).population for n in nodes]
    if all(pop == TIBETAN_POP for pop in node_pops):
        tibetan_inds.append(ind_id)

if len(tibetan_inds) == 0:
    raise ValueError("No extant Tibetan individuals found.")

tibetan_inds = sorted(tibetan_inds)

tibetan_node_pairs = []
for ind_id in tibetan_inds:
    ind = ts.individual(ind_id)
    nodes = [n for n in ind.nodes if n != tskit.NULL]
    if len(nodes) != 2:
        raise ValueError(
            f"Individual {ind_id} does not have exactly 2 nodes; found {len(nodes)}"
        )
    tibetan_node_pairs.append(nodes)

tibetan_node_pairs = np.asarray(tibetan_node_pairs, dtype=np.int32)

interval_starts = np.array([tree.interval.left for tree in ts.trees()], dtype=np.float64)
interval_ends = np.array([tree.interval.right for tree in ts.trees()], dtype=np.float64)
n_intervals = len(interval_starts)

node_population = np.array(
    [ts.node(u).population for u in range(ts.num_nodes)],
    dtype=np.int32,
)

parent_matrix = np.full((n_intervals, ts.num_nodes), -1, dtype=np.int32)
for j, tree in enumerate(ts.trees()):
    for u in range(ts.num_nodes):
        parent_matrix[j, u] = tree.parent(u)

hap0_matrix, hap1_matrix = build_haplotype_matrices_numba(
    tibetan_node_pairs,
    parent_matrix,
    node_population,
    DENISOVAN_POP,
)

n_inds = tibetan_node_pairs.shape[0]
haplotype_matrix = np.zeros((2 * n_inds, n_intervals), dtype=np.int8)
haplotype_matrix[0::2, :] = hap0_matrix
haplotype_matrix[1::2, :] = hap1_matrix

print("Haplotype matrix shape:", haplotype_matrix.shape)

if N_COLUMNS_TO_SAMPLE >= n_intervals:
    chosen_cols = np.arange(n_intervals, dtype=np.int32)
    positions = interval_starts[chosen_cols]
else:
    positions = np.linspace(
        0,
        ts.sequence_length - 1e-9,
        N_COLUMNS_TO_SAMPLE
    )
    chosen_cols = np.searchsorted(interval_ends, positions, side="right").astype(np.int32)
    chosen_cols = np.clip(chosen_cols, 0, n_intervals - 1)

print("Number of chosen columns:", len(chosen_cols))

freq_sim, mean_nonintro, mean_intro, var_nonintro, var_intro = compute_column_stats_numba(
    haplotype_matrix,
    chosen_cols,
)

positions = interval_starts[chosen_cols]
freq_values = np.full(chosen_cols.shape[0], float(freqs))

summary_df = pd.DataFrame(
    {
        "Chromosome": np.ones(len(chosen_cols), dtype=np.int32),
        "Position": positions,
        "Frequency in Tibetians": freq_values,
        "Mean non-introgressed (0) tract length": mean_nonintro,
        "Mean introgressed (1) tract length": mean_intro,
        "Variance of non-introgressed (0) tract length": var_nonintro,
        "Variance of introgressed tract length (1)": var_intro,
    }
)

print(summary_df.head())
summary_df.to_csv(OUTPUT_TABLE, index=False)
print(f"Saved table to: {OUTPUT_TABLE}")
