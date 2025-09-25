from json import load
from datetime import datetime

from pykeen.pipeline import pipeline_from_config, PipelineResult

from losses import AdversarialBCEWithoutSigmoid
from models import UVXYModel, NormModel, ProductModel, SModel
from datasets import Grid1, Grid2, Grid3, Grid4, Random1

models = (
    NormModel,
    ProductModel,
    # SModel,
)

datasets = (
    Grid1,
    Grid2,
    # Grid3,
    # Grid4,
    # Random1,
)

with open("results.tsv", "a") as f:
    f.write(f"\n\n# {datetime.now().isoformat()}\n")

for m_cls in models:
    for ds_cls in datasets:
        with open("config.json") as f: config = load(f)

        config["pipeline"]["loss"] = AdversarialBCEWithoutSigmoid
        config["pipeline"]["model"] = m_cls
        config["pipeline"]["dataset"] = ds_cls()

        result: PipelineResult = pipeline_from_config(config)

        # FIXME configs for scale/uvxy are mutually exclusive

        hits_at_1 = result.get_metric("both.realistic.hits_at_1")
        hits_at_3 = result.get_metric("both.realistic.hits_at_3")
        hits_at_10 = result.get_metric("both.realistic.hits_at_10")

        ds_name = ds_cls.name
        m_name = result.model.name

        with open("results.tsv", "a") as f:
            f.write(f"{ds_name}\t{m_name}\t{hits_at_1:.3f}\t{hits_at_3:.3f}\t{hits_at_10:.3f}\n")

        with open(f"xy_{ds_name}_{m_name}.tsv", "w") as f:
            e = result.model.entity_representations[0]()
            for x, y, ... in e: f.write(f"{x:.3f}\t{y:.3f}\n")