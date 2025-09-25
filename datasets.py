from random import random, randint, seed, shuffle
from math import log, exp, pow
from pykeen.datasets.base import PathDataset
from os.path import exists
from os import mkdir

def draw_graph(nb_entities, nb_rels, seed_nb=42):
    seed(seed_nb)

    p = 20
    max = pow(nb_entities, p)

    p_head = [ pow(i, p)/max for i in range(nb_entities) ]
    p_tail = list(p_head)

    shuffle(p_head)
    shuffle(p_tail)

    p_rel = [ 0.8 for _ in range(nb_rels) ]

    triples = set()

    for _ in range(10):
        for h, p_h in enumerate(p_head):
            for t, p_t in enumerate(p_tail):
                for r, p_r in enumerate(p_rel):
                    x_h = random()
                    x_t = random()
                    x_r = random()

                    if x_h < p_h and x_t < p_t and x_r < p_r:
                        triples.add((h, r, t))

    return list(triples)

def sample(data, nb_chunks):
    """
    Sample n items from data split in n chunks (one item per chunk)
    """
    if nb_chunks == 0 or len(data) < nb_chunks: return []

    chunk_size = len(data) // nb_chunks

    indices = [
        randint(i * chunk_size, (i+1) * chunk_size - 1)
        for i in range(nb_chunks)
    ]

    data = list(data)

    return [ data[i] for i in indices ]

def filter(data1, data2):
    return [ t for t in set(data1) - set(data2) ]

def save(filename, data):
    with open(filename, "w") as f:
        for s, p, o in data:
            f.write(f"{s}\t{p}\t{o}\n")

class GeneratedDataset(PathDataset):

    name: str

    def __init__(
            self,
            eager = False,
            create_inverse_triples = False,
            load_triples_kwargs = None
    ):
        super().__init__(
            f"data/{self.name}/train.tsv",
            f"data/{self.name}/test.tsv",
            f"data/{self.name}/valid.tsv",
            eager,
            create_inverse_triples,
            load_triples_kwargs
        )

    @classmethod
    def _generate(cls, **kwargs):
        None

    @classmethod
    def generate(cls, *, sample_ratio = 0.25, **kwargs):
        asserted, inferred = cls._generate(**kwargs)

        sample_size = int(sample_ratio * len(inferred))

        test = sample(inferred, sample_size)

        inferred = filter(inferred, test)
        valid = sample(inferred, sample_size)

        inferred = filter(inferred, valid)
        train = asserted + inferred

        if not exists("data"): mkdir("data")
        if not exists(f"data/{cls.name}"): mkdir(f"data/{cls.name}")

        save(f"data/{cls.name}/train.tsv", train)
        save(f"data/{cls.name}/valid.tsv", valid)
        save(f"data/{cls.name}/test.tsv", test)

class Grid1(GeneratedDataset):
    """
    Single rule. The order of body atoms doesn't matter.

    t(X, Z) <- r(X, Y), s(Y, Z)
    t(X, Z) <- s(X, Y), r(Y, Z)

    r: b1 = [1,0] (right translation)
    s: b2 = [0,1] (translation up)
    t: b3 = [1,1] (translation up right)
    """

    name = "grid1"

    @classmethod
    def _generate(cls, length=25):
        asserted = []
        inferred = []

        for i in range(length):
            for j in range(length):
                cell = i * length + j
                right = (i+1) * length + j
                up = i * length + (j+1)
                up_right = (i+1) * length + (j+1)

                asserted.append((cell, 0, right)) # r
                asserted.append((cell, 1, up)) # s
                
                inferred.append((cell, 2, up_right)) # t

        return asserted, inferred

class Grid2(GeneratedDataset):
    """
    Tree-structured rule base (body atoms are all distinct for a same head).

    t(X, Z) <- r(X, Y), s(Y, Z)
    t(X, Z) <- r'(X, Y), s'(Y, Z)
    
    r: w1 = [2,1]
    s: w2 = [1,2], b = [1,1]
    r': w = [2,2]
    s': b

    such that w1*w2 = w
    """

    name = "grid2"

    @classmethod
    def _generate(cls, length=25):
        asserted = []
        inferred = []

        i_max = length * length

        for i in range(length):
            for j in range(length):
                cell = i * length + j
                r_range = (2*i) * length + j
                s_range = (i+1) * length + (2*j + 1)
                rp_range = (2*i) * length + (2*j)
                sp_range = (i+1) * length + (j+1)
                t_range = (2*i + 1) * length + (2*j + 1)

                if r_range <= i_max: asserted.append((cell, 0, r_range)) # r
                if s_range <= i_max: asserted.append((cell, 1, s_range)) # s
                if rp_range <= i_max: asserted.append((cell, 2, rp_range)) # r'
                if sp_range <= i_max: asserted.append((cell, 3, sp_range)) # s'

                if t_range <= i_max: inferred.append((cell, 4, t_range)) # t

        return asserted, inferred

class Grid3(GeneratedDataset):
    """
    Regular rule base (body atoms may be repeated for a same head).

    u(X, Z) <- r(X, Y1), s(Y1, Y2), t(Y2, Z)
    u(X, Z) <- r'(X, Y1), s'(Y1, Y2), t(Y2, Z)
    
    r: w1 = [2,1]
    s: w2 = [1,4], b1 = [1,0]
    t: b = [0,1]
    r': w3 = [2,2]
    s': w4 = [1,2], b4 = [1,0]

    such that w1*w2 = w3*w4
    """

    name = "grid3"

    @classmethod
    def _generate(cls, length=25):
        asserted = []
        inferred = []

        i_max = length * length

        for i in range(length):
            for j in range(length):
                cell = i * length + j
                r_range = (2*i) * length + j
                s_range = (i+1) * length + (4*j)
                t_range = i * length + (j+1)
                rp_range = (2*i) * length + (2*j)
                sp_range = (i+1) * length + (2*j)
                u_range = (2*i + 1) * length + (4*j + 1)

                if r_range <= i_max: asserted.append((cell, 0, r_range)) # r
                if s_range <= i_max: asserted.append((cell, 1, s_range)) # s
                if t_range <= i_max: asserted.append((cell, 2, t_range)) # t
                if rp_range <= i_max: asserted.append((cell, 3, rp_range)) # r'
                if sp_range <= i_max: asserted.append((cell, 4, sp_range)) # s'

                if u_range <= i_max: inferred.append((cell, 5, u_range)) # u

        return asserted, inferred

class Grid4(GeneratedDataset):
    """
    Non-regular acyclic rule base.

    s(X, Z) <- r(X, Y), r(Y, Z)
    u(X, Z) <- t(X, Y), t(Y, Z)
    
    r: b1 = [1,0]
    s: b2 = [2,0] (two to the right)
    t: b3 = [0,1]
    u: b4 = [0,2] (two up)
    """

    name = "grid4"

    @classmethod
    def _generate(cls, length=25):
        asserted = []
        inferred = []

        for i in range(length):
            for j in range(length):
                cell = i * length + j
                right = (i+1) * length + j
                up = i * length + (j+1)
                right_right = (i+2) * length + j
                up_up = i * length + (j+2)

                asserted.append((cell, 0, right)) # r
                asserted.append((cell, 1, up)) # t
                
                inferred.append((cell, 2, right_right)) # s
                inferred.append((cell, 3, up_up)) # u

        return asserted, inferred

class Random1(GeneratedDataset):
    """
    See Grid1.
    """

    name = "random1"

    @classmethod
    def _infer(cls, triples):
        idx0 = {}
        idx1 = {}

        inferred = set()

        for h, r, t in triples:
            if r == 0:
                if t not in idx0: idx0[t] = []
                idx0[t].append(h)
            elif r == 1:
                if h not in idx1: idx1[h] = []
                idx1[h].append(t)

        for e in idx0:
            if e in idx1:
                for h in idx0[e]:
                    for t in idx1[e]:
                        inferred.add((h, 2, t))

        # TODO same for 1-0

        return list(inferred)

    @classmethod
    def _generate(cls):
        asserted = draw_graph(500, 2)
        inferred = cls._infer(asserted)

        return asserted, inferred

if __name__ == "__main__":
    Grid1.generate()
    Grid2.generate()
    Grid3.generate()
    Grid4.generate()

    Random1.generate()