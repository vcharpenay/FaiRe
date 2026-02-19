from random import random, randint, seed, shuffle
from math import log, exp, pow
from torch import tensor, cat, randperm
from pykeen.datasets.inductive import DisjointInductivePathDataset
from pykeen.triples import TriplesFactory
from os.path import exists
from os import mkdir

class CLUTTRLike:
    """
    CLUTTR-like dataset, with 'sentences' made only of compositions of triples.
    A dataset has sentence templates, defined as relational chains.
    Some sentence templates are compositional, others act as distractor.
    """

    def __init__(
        self,
        sentence_templates,
        *,
        num_sentences = 100,
        create_inverse_triples: bool = False
    ):
        rel_idx = self._collect_rels(sentence_templates)

        self.sentences = []
        i = 0

        for rel, comp_tpl, distractors in sentence_templates:
            asserted = []
            inferred = []
            not_inferred = []

            for _ in range(num_sentences // len(sentence_templates)):
                e0 = i

                i, chain, triple = self._build_triples(
                    rel_idx[rel],
                    i + 1,
                    e0,
                    [ rel_idx[rel] for rel in comp_tpl ]
                )

                asserted += chain
                inferred += [triple]

                for d_tpl in distractors:
                    i, chain, triple = self._build_triples(
                        rel_idx[rel],
                        i,
                        e0,
                        [ rel_idx[rel] for rel in d_tpl ]
                    )

                    asserted += chain
                    not_inferred += [triple]

            self.sentences.append({
                "template": (rel, comp_tpl, distractors),
                "asserted": tensor(asserted),
                "inferred": tensor(inferred),
                "not_inferred": tensor(not_inferred)
            })

        e_idx = { j: j for j in range(i) }

        # training: asserted + inferred
        # inference: asserted
        # validation: half inferred
        # test: half inferred

        # TODO separate training / inference entities?

        all_asserted = cat(list(
            sentence["asserted"]
            for sentence in self.sentences
        ))

        all_inferred = cat(list(
            sentence["inferred"]
            for sentence in self.sentences
        ))

        num_inferred = all_inferred.size(0)
        shuffling = randperm(num_inferred)

        self.training = TriplesFactory(
            mapped_triples=cat((all_asserted, all_inferred)),
            entity_to_id=e_idx,
            relation_to_id=rel_idx,
            create_inverse_triples=create_inverse_triples,
            num_entities=len(e_idx),
            num_relations=len(rel_idx)
        )

        self.inference = TriplesFactory(
            mapped_triples=all_asserted,
            entity_to_id=e_idx,
            relation_to_id=rel_idx,
            create_inverse_triples=create_inverse_triples,
            num_entities=len(e_idx),
            num_relations=len(rel_idx)
        )

        self.validation = TriplesFactory(
            mapped_triples=all_inferred[shuffling][0::2],
            entity_to_id=e_idx,
            relation_to_id=rel_idx,
            create_inverse_triples=create_inverse_triples,
            num_entities=len(e_idx),
            num_relations=len(rel_idx)
        )

        self.test = TriplesFactory(
            mapped_triples=all_inferred[shuffling][1::2],
            entity_to_id=e_idx,
            relation_to_id=rel_idx,
            create_inverse_triples=create_inverse_triples,
            num_entities=len(e_idx),
            num_relations=len(rel_idx)
        )

    def _collect_rels(self, sentence_templates):
        all_rels = set()

        for rel, comp_tpl, distractors in sentence_templates:
            all_rels |= { rel }
            all_rels |= set(comp_tpl)
            for d_tpl in distractors: all_rels |= set(d_tpl)

        # ensure reproducibility
        all_rels = list(all_rels)
        all_rels.sort()

        return { rel: i for i, rel in enumerate(all_rels) }
    
    def _build_triples(self, rel, counter, start_entity, template):
        chain = [ (start_entity, template[0], counter) ]

        for r in template[1:]:
            chain += [ (counter, r, counter + 1) ]
            counter += 1

        triple = (start_entity, rel, counter)
        counter += 1

        return counter, chain, triple

TEMPLATES = {
    "r_s_t": [
        ("t", ("r", "s"), [ ("s", "r") ])
    ],
    "r_s_t_u": [
        ("u", ("r", "s", "t"), [
            ("r", "t", "s"),
            ("s", "r", "t"),
            ("s", "t", "r"),
            ("t", "r", "s"),
            ("t", "s", "r")
        ])
    ],
    "r1_s1_r2_s2_t": [
        ("t", ("r1", "s1"), [ ("r2", "s1"), ("r1", "s2") ]),
        ("t", ("r2", "s2"), [ ("r2", "s1"), ("r1", "s2") ]),
    ],
    "r1_r2_s_t1_t2_u": [
        ("u", ("r1", "s", "t1"), [ ("r1", "s", "t2"), ("r2", "s", "t1") ]),
        ("u", ("r2", "s", "t2"), [ ("r1", "s", "t2"), ("r2", "s", "t1") ])
    ],
    "r_r_s_t": [
        ("t", ("r", "r", "s"), [ ("s"), ("r", "r", "r", "s") ]),
        ("t", ("r", "s"), [ ("s"), ("r", "r", "r", "s") ])
    ],
    "r_r_r_s_t": [
        ("t", ("r", "r", "r", "s"), [ ("r", "r", "s") ]),
        ("t", ("r", "s"), [ ("r", "r", "s") ])
    ],
    "r_s_r_t": [
        ("t", ("r", "s", "r"), [
            ("r"),
            ("s"),
            ("r", "r"),
            ("r", "r", "s"),
            ("s", "r", "r")
        ]),
        ("t", ("r", "s"), [
            ("r"),
            ("s"),
            ("r", "r"),
            ("r", "r", "s"),
            ("s", "r", "r")
        ])
    ]
}