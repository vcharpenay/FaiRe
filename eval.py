from json import load
from datetime import datetime
from random import sample

from pykeen.pipeline import pipeline_from_config, PipelineResult

from losses import AdversarialBCEWithoutSigmoid
from models import RegionBasedModel
from datasets import CLUTTRLike, TEMPLATES

def save_ranks(run: int, m: RegionBasedModel, ds: CLUTTRLike, ds_name: str, split: str):
    triples = ds.test.mapped_triples

    queries = triples[:,0:2]
    answers = triples[:,2]

    scores = m.score_t(queries)
    ranks = scores.argsort(dim=-1, descending=True)

    with open(f"ranks_{ds_name}_{m.name}.{split}.{run}.tsv", "w") as f:
        for tail, candidates in zip(answers, ranks):
            top10 = candidates[:10]

            f.write(f"{tail.item()}\t")
            f.write("\t".join([str(i.item()) for i in top10]) + "\n")

def save_r(run, m, ds_name, split):
    with open(f"r_{ds_name}_{m.name}.{split}.{run}.tsv", "w") as f:
        for rr in m.relation_representations:
            r_emb = rr()

            for r in r_emb:
                vec = r.reshape(-1)
                f.write("\t".join([str(i.item()) for i in vec]) + "\n")

            f.write("\n\n")

models = (
    # SOTA
    dict(edges = 1, scales = [-1]), # TransE
    # dict(edges = 4, scales = [-1, 0, 1, 0]), # Octagons
    # dict(edges = 2), # ExpressivE
    
    # # Polygons
    # dict(edges = 2, symmetric = False),
    # dict(edges = 4, symmetric = False),
    # dict(edges = 6, symmetric = False),
)

nb_runs = 1
# nb_runs = 3
# nb_runs = 5

for run in range(nb_runs):
    with open("results.tsv", "a") as f:
        f.write(f"\n\n# {datetime.now().isoformat()} (run #{run})\n")

    for m_config in models:
        for ds_name, template_def in TEMPLATES.items():
            with open("config.json") as f: config = load(f)

            # config["pipeline"]["training_loop"] = "LCWA"

            config["pipeline"]["loss"] = AdversarialBCEWithoutSigmoid
            config["pipeline"]["model"] = RegionBasedModel

            config["pipeline"]["model_kwargs"] |= m_config

            # config["pipeline"]["negative_sampler"] = LocalNegativeSampler

            ds = CLUTTRLike(
                sentence_templates=template_def,
                create_inverse_triples=False
            )

            num_triples = ds.training.num_triples
            indices = sample(range(num_triples), int(0.1 * num_triples))
            training_sample = ds.training.clone_and_exchange_triples(
                ds.training.mapped_triples[indices]
            )

            config["pipeline"]["training"] = ds.training
            config["pipeline"]["validation"] = training_sample
            config["pipeline"]["testing"] = training_sample

            result: PipelineResult = pipeline_from_config(config)

            save_ranks(run, result.model, ds, ds_name, "train")
            save_r(run, result.model, ds_name, "train")

            margs = config["pipeline"]["model_kwargs"]

            m_inf = RegionBasedModel(
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

            save_ranks(run, result.model, ds, ds_name, "inf")
            save_r(run, result.model, ds_name, "inf")

            # results on true triples

            hits_at_1 = result.get_metric("both.realistic.hits_at_1")
            hits_at_3 = result.get_metric("both.realistic.hits_at_3")
            hits_at_10 = result.get_metric("both.realistic.hits_at_10")

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