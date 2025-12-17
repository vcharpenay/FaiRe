from pykeen.sampling import NegativeSampler
from pykeen.sampling.filtering import PythonSetFilterer
from torch import LongTensor, long, tensor, empty, randint, stack, cat

class LocalNegativeSampler(NegativeSampler):

    def __init__(
        self,
        *,
        mapped_triples,
        num_entities = None,
        num_relations = None,
        num_negs_per_pos = None,
        filtered = False,
        filterer = PythonSetFilterer,
        filterer_kwargs = None
    ):
        super().__init__(
            mapped_triples=mapped_triples,
            num_entities=num_entities,
            num_relations=num_relations,
            num_negs_per_pos=num_negs_per_pos,
            filtered=filtered,
            filterer=filterer,
            filterer_kwargs=filterer_kwargs
        )

        components = []

        for h, _, t in mapped_triples:
            h = h.item()
            t = t.item()

            h_component = None
            t_component = None

            for c in components:
                if h in c: h_component = c
                if t in c: t_component = c

            if h_component and not t_component:
                h_component.add(t)
            elif t_component and not h_component: 
                t_component.add(h)
            elif h_component and t_component:
                if h_component != t_component:
                    components.remove(h_component)
                    components.remove(t_component)
                    components.append(h_component | t_component)
            else:
                components.append({h,t})

        max_component_size = max([ len(c) for c in components ])

        self.neighborhood_size = max_component_size
        self.index = empty((num_entities, max_component_size), dtype=long)

        for c in components:
            row = tensor(list(c))
            for e in c:
                self.index[e,:] = row
            
            # TODO padding if component isn't as large as row

    def corrupt_batch(self, positive_batch: LongTensor) -> LongTensor:
        size = positive_batch.size(0) // 2

        h = positive_batch[:size,0].unsqueeze(-1)
        t = positive_batch[size:,2].unsqueeze(-1)

        i = randint(0, self.neighborhood_size, (size, self.num_negs_per_pos))
        j = randint(0, self.neighborhood_size, (size, self.num_negs_per_pos))

        rand_t = self.index[h, i]
        rand_h = self.index[t, j]

        h = cat((h.tile((self.num_negs_per_pos,)), rand_h))
        t = cat((rand_t, t.tile((self.num_negs_per_pos,))))

        # TODO if batch size is odd, adapt size of r
        r = positive_batch[:,1].unsqueeze(-1).tile((self.num_negs_per_pos,))

        return stack((h, r, t), dim=-1)

if __name__ == "__main__":
    from datasets import Lines3

    ds = Lines3()

    s = LocalNegativeSampler(
        mapped_triples=ds.transductive_training.mapped_triples,
        num_entities=ds.transductive_training.num_entities,
        num_relations=ds.transductive_training.num_relations,
        num_negs_per_pos=5
    )

    # s = LocalNegativeSampler(
    #     mapped_triples=tensor([
    #         [0, 0, 1],
    #         [1, 1, 2],
    #         [2, 2, 3],
    #         [0, 2, 4],
    #         [4, 0, 5],
    #         [5, 1, 6],
    #         [0, 5, 3]
    #     ]),
    #     num_entities=7,
    #     num_relations=5,
    #     num_negs_per_pos=5
    # )

    b = tensor([
        [0, 0, 1],
        [1, 1, 2],
        [2, 2, 3],
        [0, 2, 4],
        [4, 0, 5],
        [5, 1, 6],
        # [0, 5, 3]
    ])

    print(s.corrupt_batch(b))