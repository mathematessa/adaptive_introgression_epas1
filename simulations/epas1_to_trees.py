import tskit
import random
import numpy as np

infile = "t-d_burnin.trees"
outfile = "t-d_burnin_with_epas1.trees"

ts = tskit.load(infile)
tables = ts.dump_tables()

DENISOVAN_POP_ID = 3

denisovan_node_ids = np.where(
    (tables.nodes.population == DENISOVAN_POP_ID) &
    (tables.nodes.time == 0)
)[0]

if len(denisovan_node_ids) == 0:
    raise ValueError("No extant Denisovan nodes found.")

node_id = random.choice(list(denisovan_node_ids))
node = tables.nodes[node_id]

mut_metadata = {
    "mutation_list": [{
        "mutation_type": 2,          # m2
        "selection_coeff": 0.03,
        "subpopulation": int(node.population),
        "slim_time": int(tables.metadata["SLiM"]["tick"] - node.time),
        "nucleotide": -1
    }]
}

# add site and mutation
site_id = tables.sites.add_row(position=2500000, ancestral_state="")
tables.mutations.add_row(
    site=site_id,
    node=node_id,
    derived_state="1",
    time=node.time,
    metadata=mut_metadata
)

new_ts = tables.tree_sequence()
new_ts.dump(outfile)
print(f"Wrote {outfile}")
print(f"Added EPAS1 mutation to node {node_id} in Denisovan population {DENISOVAN_POP_ID}")
