from pandas import read_csv

nb_runs = 1
# nb_runs = 3

series = [(0, 50), (52, 50)]
# series = [(0,50)]
# series = [(52,50)]

# models = [ "u_norm", "v_norm", "uv_norm", "x_scale", "xy_scale" ]
# datasets = [ "rst", "rstu_1", "rstu_2", "rrst" ]

# models = [ "uvxy_norm", "xyw_scale", "xy_scale", "1_poly", "2_poly", "3_poly" ]
# datasets = [ "rstu_2_full" ]

models = [ "uvxy_norm", "xy_scale", "2_poly" ]
datasets = [ "rrst_full" ]

for m in models:
    for ds in datasets:
        precision_per_run = []

        for run in range(nb_runs):
            precision_per_series = []

            for s in series:
                df = read_csv(
                    f"scores_{ds}_{m}.{run}.tsv",
                    sep="\t",
                    header=None,
                    skiprows=s[0],
                    nrows=s[1]
                )

                scores = df.to_numpy()[:,1:]

                precision = (scores[:,0:1] > scores[:,1:]).all(1).mean()
                
                precision_per_series.append(precision)
                
            p = sum(precision_per_series) / len(series)
            precision_per_run.append(f"{p:.3f}")

        print(f"{m}\t{ds}\t{"\t".join(precision_per_run)}")