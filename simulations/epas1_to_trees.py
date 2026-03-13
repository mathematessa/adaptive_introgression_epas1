import tskit
import pyslim
import numpy as np

infile = "t-d_burnin.trees"
outfile = "t-d_burnin_with_epas1.trees"

ts = tskit.load(infile)
tables = ts.dump_tables()

# 0 ancestral_hominin
# 1 denisovan
# 2 modern_human
# 3 african
# 4 non_african
# 5 eurasian
# 6 tibetan
DENISOVAN_POP_ID = 1
POS = 2500000

# extant Denisovan nodes
den_nodes = np.where(
    (tables.nodes.population == DENISOVAN_POP_ID) &
    (tables.nodes.time == 0)
)[0]

if len(den_nodes) == 0:
    raise ValueError("No extant Denisovan nodes found.")

# find the tree covering the EPAS1 position
tree = ts.at(POS)
mrca = den_nodes[0]
for u in den_nodes[1:]:
    mrca = tree.mrca(mrca, u)
    if mrca == tskit.NULL:
        raise ValueError("Denisovan nodes do not share an MRCA at this position.")

node = tables.nodes[mrca]

existing_ids = []
for mut in ts.mutations():
    ds = mut.derived_state.strip()
    if ds != "":
        existing_ids.extend(int(x) for x in ds.split(","))
new_mut_id = max(existing_ids, default=0) + 1

mut_metadata = {
    "mutation_list": [{
        "mutation_type": 2,            # m2
        "selection_coeff": 0.03,
        "subpopulation": int(node.population),
        "slim_time": int(tables.metadata["SLiM"]["tick"] - node.time),
        "nucleotide": -1
    }]
}

site_id = tables.sites.add_row(position=POS, ancestral_state="")

tables.mutations.add_row(
    site=site_id,
    node=mrca,
    derived_state=str(new_mut_id),
    time=node.time,
    metadata=mut_metadata
)

new_ts = tables.tree_sequence()
new_ts.dump(outfile)
print(f"Wrote {outfile}")
print(f"Added EPAS1 at position {POS} on MRCA node {mrca}")
print(f"SLiM mutation ID: {new_mut_id}")
