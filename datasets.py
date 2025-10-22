from random import random, randint, seed, shuffle
from math import log, exp, pow
from pykeen.datasets.base import PathDataset
from pykeen.datasets.inductive import DisjointInductivePathDataset
from os.path import exists
from os import mkdir

# TODO draw graph as tensor?

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

class GeneratedDataset(DisjointInductivePathDataset):
    """
    The dataset has two separate components: G1, G2.
    The training split contains G1 and all asserted triples of G2.
    The validation and test splits contain the inferred triples of G2.
    """

    name: str

    def __init__(
            self,
            eager = False,
            create_inverse_triples = False,
            load_triples_kwargs = None
    ):
        super().__init__(
            f"data/{self.name}/train.tsv",
            f"data/{self.name}/inf.tsv",
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
    def generate(cls, **kwargs):
        g1, g2_asserted, g2_inferred = cls._generate(**kwargs)

        shuffle(g2_inferred)

        train = g1
        inf = g2_asserted
        valid = g2_inferred[::2]
        test = g2_inferred[1::2]

        if not exists("data"): mkdir("data")
        if not exists(f"data/{cls.name}"): mkdir(f"data/{cls.name}")

        save(f"data/{cls.name}/train.tsv", train)
        save(f"data/{cls.name}/inf.tsv", inf)
        save(f"data/{cls.name}/valid.tsv", valid)
        save(f"data/{cls.name}/test.tsv", test)

class Grid(GeneratedDataset):

    test_factor = 1

    @classmethod
    def _generate_cell(cls, i, j, length):
        None
    
    @classmethod
    def _index(cls, triples, length, offset):
        return [
            (i * length + j + offset, r, k * length + l + offset)
            for ((i,j), r, (k,l)) in triples
            if i < length and j < length and k < length and l < length
        ]

    @classmethod
    def _generate(cls, length=10):
        g1 = []
        g2_asserted = []
        g2_inferred = []

        g1_length = length
        g2_length = int(length * cls.test_factor)

        for i in range(g1_length):
            for j in range(g1_length):
                asserted, inferred = cls._generate_cell(i, j)
                g1 += cls._index(asserted, g1_length, 0)
                g1 += cls._index(inferred, g1_length, 0)

        offset = g1_length * g1_length

        for i in range(g2_length):
            for j in range(g2_length):
                asserted, inferred = cls._generate_cell(i, j)
                g2_asserted += cls._index(asserted, g2_length, offset)
                g2_inferred += cls._index(inferred, g2_length, offset)

        return g1, g2_asserted, g2_inferred

class Grid1(Grid):
    """
    Single rule. The order of body atoms doesn't matter.

    t(X, Z) <- r(X, Y), s(Y, Z)
    t(X, Z) <- s(X, Y), r(Y, Z)

    r: b1 = [1,0] (right translation)
    s: b2 = [0,1] (translation up)
    t: b3 = [1,1] (translation up right)
    """

    name = "grid1"

    # test_factor = 0.5

    @classmethod
    def _generate_cell(cls, i, j):
        cell = (i, j)
        right = (i+1, j)
        up = (i, j+1)
        up_right = (i+1, j+1)
        
        return [
            (cell, 0, right), # r
            (cell, 1, up) # s
        ], [
            (cell, 2, up_right) # t
        ]

class Grid2(Grid):
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

    # test_factor = 1

    @classmethod
    def _generate_cell(cls, i, j):
        cell = (i, j)
        r_range = (2*i, j)
        s_range = (i+1, 2*j + 1)
        rp_range = (2*i, 2*j)
        sp_range = (i+1, j+1)
        t_range = (2*i + 1, 2*j + 1)

        return [
            (cell, 0, r_range), # r
            (cell, 1, s_range), # s
            (cell, 2, rp_range), # r'
            (cell, 3, sp_range) # s'
        ], [
            (cell, 4, t_range) # t
        ]

class Grid3(Grid):
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

    # test_factor = 2

    @classmethod
    def _generate_cell(cls, i, j):
        cell = (i, j)
        r_range = (2*i, j)
        s_range = (i+1, 4*j)
        t_range = (i, j+1)
        rp_range = (2*i, 2*j)
        sp_range = (i+1, 2*j)
        u_range = (2*i + 1, 4*j + 1)

        return [
            (cell, 0, r_range), # r
            (cell, 1, s_range), # s
            (cell, 2, t_range), # t
            (cell, 3, rp_range), # r'
            (cell, 4, sp_range) # s'
        ], [
            (cell, 5, u_range) # u
        ]

class Grid4(Grid):
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

    # test_factor = 0.5

    @classmethod
    def _generate_cell(cls, i, j):
        cell = (i, j)
        right = (i+1, j)
        up = (i, j+1)
        right_right = (i+2, j)
        up_up = (i, j+2)

        return [
            (cell, 0, right), # r
            (cell, 1, up) # t
        ], [
            (cell, 2, right_right), # s
            (cell, 3, up_up) # u
        ]

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

    # ratio = 0.25 # TODO as argument

    # Random1.generate(sample_ratio=ratio)