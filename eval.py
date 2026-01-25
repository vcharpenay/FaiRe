from json import load
from datetime import datetime

from pykeen.pipeline import pipeline_from_config, PipelineResult
from pykeen.evaluation import RankBasedEvaluator

from losses import BCEWithoutSigmoid, AdversarialBCEWithoutSigmoid
from models import UVXYModel, NormModel, ProductModel, SModel, SWModel, PolygonModel, FFNModel
from datasets import Grid1, Grid2, Grid3, Grid4, Lines1, Lines2, Lines3, Random1, CLUTTRLike
from sampling import LocalNegativeSampler

def save_xy(run, m, ds_name, split):
    e_emb = m.entity_representations[0]()

    if e_emb.size(-1) == 2:
        with open(f"xy_{ds_name}_{m.name}.{split}.{run}.tsv", "w") as f:
            for e in e_emb:
                x = e[0]
                y = e[1]
                f.write(f"{x:.3f}\t{y:.3f}\n")

def save_r(run, m, ds_name, split):
    with open(f"r_{ds_name}_{m.name}.{split}.{run}.tsv", "w") as f:
        for rr in m.relation_representations:
            r_emb = rr()

            for r in r_emb:
                vec = r.reshape(-1)
                f.write("\t".join([str(i.item()) for i in vec]) + "\n")

            f.write("\n\n")

models = (
    # NormModel,
    # ProductModel,
    # SModel,
    # SWModel,
    PolygonModel,
    # FFNModel,
)

datasets = (
    # Grid1,
    # Grid2,
    # Grid3,
    # Grid4,
    # Lines1,
    # Lines2,
    # Lines3,
    # Random1,
)

templates = {
    # "rst": [
    #     ("t", ("r", "s"), [ ("s", "r") ])
    # ],
    # "rstu_1": [
    #     ("u", ("r", "s", "t"), [ ("t", "r", "s") ]),
    #     ("u", ("r'", "s'", "t"), [ ("t", "r'", "s'") ])
    # ],
    # "rstu_2": [
    #     ("u", ("r", "s", "t"), [ ("t", "s", "r") ]),
    #     ("u", ("r'", "s'", "t"), [ ("t", "s'", "r'") ])
    # ],
    # "rstu_2_full": [
    #     (
    #         "u",
    #         ("r", "s", "t"),
    #         [
    #             ("r",),
    #             ("s",),
    #             ("t",),
    #             ("r", "s"),
    #             ("s", "r"),
    #             ("r", "t"),
    #             ("t", "r"),
    #             ("s", "t"),
    #             ("t", "s"),
    #             ("r", "t", "s"),
    #             ("s", "r", "t"),
    #             ("s", "t", "r"),
    #             ("t", "s", "r"),
    #             ("t", "r", "s")
    #         ]
    #     ),
    #     (
    #         "u",
    #         ("r'", "s'", "t"),
    #         [
    #             ("r'", "s", "t"),
    #             ("r", "s'", "t")
    #         ]
    #     )
    # ],
    # "rrst": [
    #     ("t", ("r", "r", "s"), [ ("s",), ("s", "r") ]),
    #     ("t", ("r", "s"), [ ("s",), ("s", "r") ])
    # ],
    "rrst_full": [
        (
            "t",
            ("r", "r", "s"),
            [
                ("s",),
                ("s", "r"),
                ("s", "s", "r"),
                ("s", "r", "r"),
                ("r", "s", "r"),
                ("r", "r", "r", "s")
            ]
        ),
        (
            "t",
            ("r", "s"),
            [
                ("s",),
                ("s", "r"),
                ("s", "s", "r"),
                ("s", "r", "r"),
                ("r", "s", "r"),
                ("r", "r", "r", "s")
            ]
        )
    ]
}

nb_runs = 1
# nb_runs = 3

for run in range(nb_runs):
    with open("results.tsv", "a") as f:
        f.write(f"\n\n# {datetime.now().isoformat()} (run #{run})\n")

    for m_cls in models:
        for ds_name, template_def in templates.items():
            with open("config.json") as f: config = load(f)

            config["pipeline"]["loss"] = AdversarialBCEWithoutSigmoid
            config["pipeline"]["model"] = m_cls

            # config["pipeline"]["negative_sampler"] = LocalNegativeSampler

            ds = CLUTTRLike(
                sentence_templates=template_def,
                create_inverse_triples=False
            )

            # TODO validation on a small subset / on inferrable triples?
            config["pipeline"]["training"] = ds.training
            config["pipeline"]["validation"] = ds.training
            config["pipeline"]["testing"] = ds.training

            result: PipelineResult = pipeline_from_config(config)

            save_xy(run, result.model, ds_name, "train")
            save_r(run, result.model, ds_name, "train")

            # FIXME configs for scale/uvxy are mutually exclusive

            margs = config["pipeline"]["model_kwargs"]

            m_inf = m_cls(
                triples_factory=ds.inference,
                r_pretrained=result.model.relation_representations,
                loss=AdversarialBCEWithoutSigmoid,
                **margs
            )

            # no fine-tuning
            for rr in m_inf.relation_representations: rr.requires_grad_(False)

            config["pipeline"]["model"] = m_inf

            config["pipeline"]["training"] = ds.inference
            config["pipeline"]["validation"] = ds.validation
            config["pipeline"]["testing"] = ds.test

            result: PipelineResult = pipeline_from_config(config)

            save_xy(run, result.model, ds_name, "inf")
            save_r(run, result.model, ds_name, "inf")

            # results on true triples

            hits_at_1 = result.get_metric("both.realistic.hits_at_1")
            hits_at_3 = result.get_metric("both.realistic.hits_at_3")
            hits_at_10 = result.get_metric("both.realistic.hits_at_10")

            # note: validation on "inverse_harmonic_mean_rank"?

            with open("results.tsv", "a") as f:
                f.write(f"{ds_name}\t{result.model.name}\t{hits_at_1:.3f}\t{hits_at_3:.3f}\t{hits_at_10:.3f}\n")

            with open(f"scores_{ds_name}_{result.model.name}.{run}.tsv", "w") as f:
                for data in ds.sentences:
                    rel, comp_tpl, distractors = data["template"]
                    
                    pos = m_inf.score_hrt(data["inferred"])
                    neg = m_inf.score_hrt(data["not_inferred"])

                    neg = neg.reshape((-1, len(distractors)))

                    for pos_score, neg_scores in zip(pos, neg):
                        p = pos_score.item()
                        n = "\t".join([
                            str(score)
                            for score in neg_scores.tolist()
                        ])

                        f.write(f"{rel}\t{p}\t{n}\n")
                            
                    f.write("\n\n")