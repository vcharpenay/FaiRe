from torch import prod, cat
from torch.linalg import vector_norm
from torch.nn.init import xavier_uniform_

from pykeen.nn import Embedding
from pykeen.nn.modules import Interaction
from pykeen.models import ERModel
    
def filter(constraints, a):
    d = a.size(-2) // 4

    if "u" not in constraints: a[..., :d, :] = float("-inf")
    if "v" not in constraints: a[..., d:2*d, :] = float("-inf")
    if "x" not in constraints: a[..., 2*d:3*d, :] = float("-inf")
    if "y" not in constraints: a[..., 3*d:, :] = float("-inf")

    return a

class NormInteraction(Interaction):

    def __init__(
        self,
        *,
        n = 2,
        constraints = "uvxy",
    ):
        super().__init__()

        self.n = n
        self.constraints = constraints

        self.entity_shape = ("d",)
        self.relation_shape = (("e", "d"), ("e", "d"), ("e", "d"))

    def forward(self, h, r, t):
        c, _, a = r

        a = filter(self.constraints, a).softmax(dim=-2)

        cu, cv, cx, cy = c.tensor_split(4, dim=-2)
        au, av, ax, ay = a.tensor_split(4, dim=-2)

        u = (h - t).unsqueeze(-2)
        v = (h + t).unsqueeze(-2)
        x = h.unsqueeze(-2)
        y = t.unsqueeze(-2)

        dist_u = au * (u + cu)
        dist_v = av * (v + cv)
        dist_x = ax * (x + cx)
        dist_y = ay * (y + cy)

        dist = (dist_u + dist_v + dist_x + dist_y).sum(dim=-2)

        return (1 - vector_norm(dist, dim=-1)).sigmoid()

class ProductInteraction(Interaction):

    def __init__(
        self,
        *,
        n = 2,
        constraints = "uvxy",
    ):
        super().__init__()

        self.entity_shape = ("d",)
        self.relation_shape = (("e", "d"), ("e", "d"), ("e", "d"))

        self.n = n
        self.constraints = constraints

    def forward(self, h, r, t):
        c, w, a = r

        w = abs(w)
        a = filter(self.constraints, a).softmax(dim=-2)

        cu, cv, cx, cy = c.tensor_split(4, dim=-2)
        wu, wv, wx, wy = w.tensor_split(4, dim=-2)
        au, av, ax, ay = a.tensor_split(4, dim=-2)

        u = (h - t).unsqueeze(-2)
        v = (h + t).unsqueeze(-2)
        x = h.unsqueeze(-2)
        y = t.unsqueeze(-2)

        pu = au * self.gaussian(u, cu, wu)
        pv = av * self.gaussian(v, cv, wv)
        px = ax * self.gaussian(x, cx, wx)
        py = ay * self.gaussian(y, cy, wy)

        p = (pu + pv + px + py).sum(dim=-2)

        # FIXME returned values not in [0,1]?
        return p.prod(dim=-1)
    
    def gaussian(self, e, c, w):
        return (-((e - c)/w)**2).exp()

class UVXYModel(ERModel):

    def __init__(
        self,
        *,
        interaction: Interaction,
        embedding_dim: int = 40,
        n: int = 2,
        constraints: str = "uvxy",
        scales: int = "x",
        **kwargs
    ) -> None:
        e_kwargs = dict(
            embedding_dim=embedding_dim,
            initializer=xavier_uniform_
        )

        r_kwargs = [
            dict( # center
                shape=(n*4, embedding_dim),
                # FIXME init should be per component
                initializer=xavier_uniform_
            ),
            dict( # width
                shape=(n*4, embedding_dim)
            ),
            dict( # attention weight
                shape=(n*4, embedding_dim)
            ),
        ]

        super().__init__(
            interaction=interaction,
            interaction_kwargs=dict(
                n=n,
                constraints=constraints,
            ),
            entity_representations=Embedding,
            entity_representations_kwargs=e_kwargs,
            relation_representations=Embedding,
            relation_representations_kwargs=r_kwargs,
            **kwargs
        )

class NormModel(UVXYModel):

    def __init__(
        self,
        *,
        constraints: str = "uvxy",
        **kwargs
    ) -> None:
        super().__init__(
            interaction=NormInteraction,
            constraints=constraints,
            **kwargs
        )

        self.name = constraints + "_norm"

class ProductModel(UVXYModel):

    def __init__(
        self,
        *,
        constraints: str = "uvxy",
        **kwargs
    ) -> None:
        super().__init__(
            interaction=ProductInteraction,
            constraints=constraints,
            **kwargs
        )

        self.name = constraints + "_prod"

class SInteraction(Interaction):

    def __init__(
            self,
            *,
            scales = "x",
    ):
        super().__init__()

        self.entity_shape = ("d",)
        self.relation_shape = (("e", "d"), ("e", "d"))

        self.scales = scales

    def forward(self, h, r, t):
        s, b = r

        sx, sy = s.tensor_split(2, dim=-2)
        bx, by = b.tensor_split(2, dim=-2)

        x = h.unsqueeze(-2)
        y = t.unsqueeze(-2)

        dist_x = sx * x + y - bx
        dist_y = sy * y + x - by

        dist_cat = []
        if "x" in self.scales: dist_cat.append(dist_x)
        if "y" in self.scales: dist_cat.append(dist_y)

        dist = cat(dist_cat, dim=-1).squeeze(-2)

        return (1 - vector_norm(dist, dim=-1)).sigmoid()
    
    def filter(self, scales, a):
        if "x" not in scales: a[...,0,:] = float("-inf")
        if "y" not in scales: a[...,1,:] = float("-inf")

        return a

class SModel(ERModel):

    def __init__(
        self,
        *,
        embedding_dim: int = 40,
        scales: int = "x",
        n: int = 2,
        constraints: str = "uvxy",
        **kwargs
    ) -> None:
        e_kwargs = dict(
            embedding_dim=embedding_dim,
            initializer=xavier_uniform_
        )

        r_kwargs = [
            dict( # scale
                shape=(2, embedding_dim)
            ),
            dict( # offset
                shape=(2, embedding_dim),
                # FIXME init should be per component
                initializer=xavier_uniform_
            ),
        ]

        super().__init__(
            interaction=SInteraction,
            interaction_kwargs=dict(
                scales=scales,
            ),
            entity_representations=Embedding,
            entity_representations_kwargs=e_kwargs,
            relation_representations=Embedding,
            relation_representations_kwargs=r_kwargs,
            **kwargs
        )

        self.name = scales + "_scale"