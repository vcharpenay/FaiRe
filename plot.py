from math import sqrt
from statistics import mean, stdev
from pandas import read_csv
from matplotlib.pyplot import subplots

from datasets import TEMPLATES

# nb_runs = 1
# nb_runs = 3
# nb_runs = 5
nb_runs = 20

nb_instances = 50

models = [
    "1_sym_s_model",
    "4_sym_s_model",
    "2_sym_model",
    "2_model",
    "4_model",
    "6_model",
    "8_model",
    "10_model"
]

model_names = [
    "TransE",
    "Octagons",
    "ExpressivE",
    "Polygons (2)",
    "Polygons (4)",
    "Polygons (6)",
    "Polygons (8)",
    "Polygons (10)"
]

ds_mapping = {
    "r_s_t": "Perm2",
    "r_s_t_u": "Perm3",
    "r1_s1_r2_s2_t": "Mix2",
    "r1_r2_s_t1_t2_u": "Mix3",
    "r_r_s_t": "Rep12",
    "r_r_r_s_t": "Rep13",
    "r_s_r_t": "Comb"
}

x_pos = [0, 1, 2, 4, 5, 6, 7, 8] # leave space between polygons and others

for ds, tpl in TEMPLATES.items():
    fig, ax = subplots()

    precision_per_model = []
    min_precision_per_model = []
    max_precision_per_model = []

    for m_i, m in enumerate(models):
        precision_per_run = []

        for run in range(nb_runs):
            precision_per_series = []

            for i, _ in enumerate(tpl):
                df = read_csv(
                    f"scores_{ds}_{m}.{run}.tsv",
                    sep="\t",
                    header=None,
                    skiprows=i * (nb_instances + 2),
                    nrows=nb_instances
                )

                scores = df.to_numpy()[:,1:]

                precision = (scores[:,0:1] > scores[:,1:]).all(1).mean()
                
                precision_per_series.append(precision.item())
                
            p = sum(precision_per_series) / len(tpl)
            precision_per_run.append(p)

        p_avg = mean(precision_per_run)
        precision_per_model.append(p_avg)

        p_min = min(precision_per_run)
        p_max = max(precision_per_run)
        
        if nb_runs > 1:
            p_stdev = stdev(precision_per_run)
            min_precision_per_model.append(p_stdev)
            max_precision_per_model.append(p_stdev)
        else:
            min_precision_per_model.append(p_avg - p_min)
            max_precision_per_model.append(p_max - p_avg)

        ds_name = ds_mapping[ds]
        m_name = model_names[m_i]
        print(f"{ds_name}\t{m_name}\t{p_avg:.3f}\t{p_min:.3f}\t{p_max:.3f}")

    err = [min_precision_per_model, max_precision_per_model]

    ax.bar(x_pos, precision_per_model, yerr=err)
    ax.set_xticks(x_pos, labels=model_names, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ybound(0, 1)
    ax.set_ylabel('hits@1')
    ax.set_title(ds_mapping[ds])

    fig.tight_layout()
    fig.savefig(f"plot_{ds}.pdf")
    
    # FIXME dataset names should be aligned with template names