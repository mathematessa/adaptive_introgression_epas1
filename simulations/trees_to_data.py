import pyslim
import tskit
import numpy as np
import pandas as pd

INPUT_FILE = "epas1_introgression_output.trees"
OUTPUT_FILE = "synthetic_introgression_table.csv"
ts = tskit.load(INPUT_FILE)

TIBETAN = 6
DENISOVAN = 1

tibetan_samples = ts.samples(population=TIBETAN)
denisovan_samples = set(ts.samples(population=DENISOVAN))

tracts = []

for sample in tibetan_samples:
    current_start = None
    for tree in ts.trees():
        left, right = tree.interval
        node = sample
        is_denisovan = False
        while node != tskit.NULL:
            if node in denisovan_samples:
                is_denisovan = True
                break
            node = tree.parent(node)

        if is_denisovan:
            if current_start is None:
                current_start = left
        else:
            if current_start is not None:
                tracts.append(right - current_start)
                current_start = None

    if current_start is not None:
        tracts.append(ts.sequence_length - current_start)

tracts = np.array(tracts)
mean_tract = np.mean(tracts) if len(tracts) > 0 else 0
var_tract = np.var(tracts) if len(tracts) > 0 else 0


records = []
for tree in ts.trees():
    left, right = tree.interval
    den_count = 0
    for sample in tibetan_samples:
        node = sample
        while node != tskit.NULL:
            if node in denisovan_samples:
                den_count += 1
                break
            node = tree.parent(node)
    freq = den_count / len(tibetan_samples)
    records.append({
        "Chromosome": 0,
        "Position": left,
        "Frequency_Archaic_Pop0": freq,
        "Mean_Tract_Pop0": mean_tract,
        "Mean_Tract_Pop1": mean_tract,
        "Var_Tract_Pop0": var_tract,
        "Var_Tract_Pop1": var_tract
    })

df = pd.DataFrame(records)
df.to_csv(OUTPUT_FILE, index=False)

print("Dataset saved to:", OUTPUT_FILE)
print(df.head())
