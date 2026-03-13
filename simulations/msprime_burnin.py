import msprime
import pyslim

sequence_length = 5e6
recombination_rate = 1e-8

Ne_ancestral = 20000
Ne_modern = 20000
Ne_african = 30000
Ne_non_african = 5000
Ne_eurasian = 8000
Ne_tibetan = 10000
Ne_denisovan = 3000

T_DENISOVAN_SPLIT = 20000   # ~500 kya
T_OUT_OF_AFRICA = 2800      # ~70 kya
T_EURASIAN_TIBETAN = 1200   # ~30 kya

n_african = 200
n_eurasian = 200
n_tibetan = 200
n_denisovan = 50

demography = msprime.Demography()

demography.add_population(
    name="ancestral_hominin",
    initial_size=Ne_ancestral
)

demography.add_population(
    name="denisovan",
    initial_size=Ne_denisovan
)

demography.add_population(
    name="modern_human",
    initial_size=Ne_modern
)

# modern human structure
demography.add_population(
    name="african",
    initial_size=Ne_african
)

demography.add_population(
    name="non_african",
    initial_size=Ne_non_african
)

demography.add_population(
    name="eurasian",
    initial_size=Ne_eurasian
)

demography.add_population(
    name="tibetan",
    initial_size=Ne_tibetan
)

# Denisovan split
demography.add_population_split(
    time=T_DENISOVAN_SPLIT,
    derived=["modern_human", "denisovan"],
    ancestral="ancestral_hominin"
)

# out of Africa
demography.add_population_split(
    time=T_OUT_OF_AFRICA,
    derived=["african", "non_african"],
    ancestral="modern_human"
)

# Eurasian / Tibetan split
demography.add_population_split(
    time=T_EURASIAN_TIBETAN,
    derived=["eurasian", "tibetan"],
    ancestral="non_african"
)

samples = [
    msprime.SampleSet(n_african, population="african", time=0),
    msprime.SampleSet(n_eurasian, population="eurasian", time=0),
    msprime.SampleSet(n_tibetan, population="tibetan", time=0),
    msprime.SampleSet(n_denisovan, population="denisovan", time=0),
]

demography.sort_events()

ts = msprime.sim_ancestry(
    samples=samples,
    demography=demography,
    recombination_rate=recombination_rate,
    sequence_length=sequence_length,
    # record_migrations=True
)

ts_slim = pyslim.annotate(
    ts,
    model_type="nonWF",
    tick=1
)

ts_slim.dump("t-d_burnin.trees")

print("Tree sequence generated successfully.")
print("Individuals:", ts_slim.num_individuals)
print("Trees:", ts_slim.num_trees)
