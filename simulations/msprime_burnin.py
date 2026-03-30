import msprime
import pyslim

sequence_length = 300_000
recombination_rate = 1.25e-8

Ne_ancestral = 10_000
Ne_modern = 20_000
Ne_african = 10_000
Ne_non_african = 5_000
Ne_eurasian = 10_000
Ne_tibetan = 7_000
Ne_denisovan = 10_000

T_DENISOVAN_SPLIT = 20_000   # ~500 kya 25 yr/gen
T_OUT_OF_AFRICA = 2_800      # ~70 kya
T_EURASIAN_TIBETAN = 1_200   # ~30 kya

n_african = 1000
n_eurasian = 1000
n_tibetan = 7000
n_denisovan = 1000

demography = msprime.Demography()

demography.add_population(name="ancestral_hominin", initial_size=Ne_ancestral)
demography.add_population(name="denisovan", initial_size=Ne_denisovan)
demography.add_population(name="modern_human", initial_size=Ne_modern)
demography.add_population(name="african", initial_size=Ne_african)
demography.add_population(name="non_african", initial_size=Ne_non_african)
demography.add_population(name="eurasian", initial_size=Ne_eurasian)
demography.add_population(name="tibetan", initial_size=Ne_tibetan)

demography.add_population_split(
    time=T_DENISOVAN_SPLIT,
    derived=["modern_human", "denisovan"],
    ancestral="ancestral_hominin",
)

demography.add_population_split(
    time=T_OUT_OF_AFRICA,
    derived=["african", "non_african"],
    ancestral="modern_human",
)

demography.add_population_split(
    time=T_EURASIAN_TIBETAN,
    derived=["eurasian", "tibetan"],
    ancestral="non_african",
)

demography.sort_events()

samples = [
    msprime.SampleSet(n_african, population="african", time=0),
    msprime.SampleSet(n_eurasian, population="eurasian", time=0),
    msprime.SampleSet(n_tibetan, population="tibetan", time=0),
    msprime.SampleSet(n_denisovan, population="denisovan", time=0),
]

ts = msprime.sim_ancestry(
    samples=samples,
    demography=demography,
    recombination_rate=recombination_rate,
    sequence_length=sequence_length,
    random_seed=42,
)

ts_slim = pyslim.annotate(ts, model_type="WF", tick=1)
ts_slim.dump("/Users/ekaterina/Desktop/RESEARCH/SLiM/t_d_burnin3.trees")

print("Wrote: t_d_burnin.trees")
print("num_individuals =", ts_slim.num_individuals)
print("num_trees       =", ts_slim.num_trees)

for pop in ts_slim.populations():
    md = pop.metadata
    name = md["name"] if md is not None and "name" in md else "N/A"
    print(f"population_id={pop.id}, name={name}")
