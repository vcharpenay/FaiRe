from json import load
from datetime import datetime

from pykeen.pipeline import pipeline_from_config, PipelineResult
from pykeen.evaluation import RankBasedEvaluator

from losses import BCEWithoutSigmoid, AdversarialBCEWithoutSigmoid
from models import UVXYModel, NormModel, ProductModel, SModel
from datasets import Grid1, Grid2, Grid3, Grid4, Lines1, Lines2, Lines3, Random1
from sampling import LocalNegativeSampler

def save_xy(m, ds_name, split):
    e_emb = m.entity_representations[0]()

    if e_emb.size(-1) == 2:
        with open(f"xy_{ds_name}_{m.name}.{split}.tsv", "w") as f:
            for e in e_emb:
                x = e[0]
                y = e[1]
                f.write(f"{x:.3f}\t{y:.3f}\n")

def save_r(m, ds_name, split):
    with open(f"r_{ds_name}_{m.name}.{split}.tsv", "w") as f:
        for rr in m.relation_representations:
            r_emb = rr()

            for r in r_emb:
                vec = r.reshape(-1)
                f.write("\t".join([str(i.item()) for i in vec]) + "\n")

            f.write("\n\n")

models = (
    # NormModel,
    ProductModel,
    # SModel,
)

datasets = (
    # Grid1,
    # Grid2,
    # Grid3,
    # Grid4,
    # Lines1,
    Lines2,
    # Lines3,
    # Random1,
)

with open("results.tsv", "a") as f:
    f.write(f"\n\n# {datetime.now().isoformat()}\n")

for m_cls in models:
    for ds_cls in datasets:
        with open("config.json") as f: config = load(f)

        ds_name = ds_cls.name

        config["pipeline"]["loss"] = AdversarialBCEWithoutSigmoid
        config["pipeline"]["model"] = m_cls

        # config["pipeline"]["negative_sampler"] = LocalNegativeSampler

        ds = ds_cls(create_inverse_triples=True)
        # TODO validation on a small subset / on inferrable triples?
        config["pipeline"]["training"] = ds.transductive_training
        config["pipeline"]["validation"] = ds.transductive_training
        config["pipeline"]["testing"] = ds.transductive_training

        result: PipelineResult = pipeline_from_config(config)

        save_xy(result.model, ds_name, "train")
        save_r(result.model, ds_name, "train")

        # FIXME configs for scale/uvxy are mutually exclusive

        margs = config["pipeline"]["model_kwargs"]

        m_inf = m_cls(
            triples_factory=ds.inductive_inference,
            r_pretrained=result.model.relation_representations,
            loss=AdversarialBCEWithoutSigmoid,
            **margs
        )

        # no fine-tuning
        for rr in m_inf.relation_representations: rr.requires_grad_(False)

        config["pipeline"]["model"] = m_inf

        config["pipeline"]["training"] = ds.inductive_inference
        config["pipeline"]["validation"] = ds.inductive_validation
        config["pipeline"]["testing"] = ds.inductive_testing

        result: PipelineResult = pipeline_from_config(config)

        save_xy(result.model, ds_name, "inf")
        save_r(result.model, ds_name, "inf")

        # results on true triples

        hits_at_1 = result.get_metric("both.realistic.hits_at_1")
        hits_at_3 = result.get_metric("both.realistic.hits_at_3")
        hits_at_10 = result.get_metric("both.realistic.hits_at_10")

        # note: validation on "inverse_harmonic_mean_rank"?

        with open("results.tsv", "a") as f:
            f.write(f"{ds_name}\t{result.model.name}\t{hits_at_1:.3f}\t{hits_at_3:.3f}\t{hits_at_10:.3f}\n")

        # results on false triples

        # evaluator = RankBasedEvaluator()
        # result = evaluator.evaluate(
        #     model=m_inf,
        #     mapped_triples=ds.inductive_testing_neg,
        # )

        # hits_at_1 = result.get_metric("both.realistic.hits_at_1")
        # hits_at_3 = result.get_metric("both.realistic.hits_at_3")
        # hits_at_10 = result.get_metric("both.realistic.hits_at_10")

        # with open("results.tsv", "a") as f:
        #     f.write(f"{ds_name} (neg)\t{result.model.name}\t{hits_at_1:.3f}\t{hits_at_3:.3f}\t{hits_at_10:.3f}\n")