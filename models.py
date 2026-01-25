from collections.abc import Sequence

from torch import tensor, prod, cat, where, FloatTensor
from torch.linalg import vector_norm
from torch.nn.functional import relu, gelu, leaky_relu, softplus
from torch.nn.init import xavier_uniform_, uniform_, normal_

from pykeen.nn import Embedding
from pykeen.nn.modules import Interaction
from pykeen.nn.representation import Representation
from pykeen.models import ERModel
    
def filter(constraints, a):
    d = a.size(-2) // 4

    if "u" not in constraints: a[..., :d, :] = float("-inf")
    if "v" not in constraints: a[..., d:2*d, :] = float("-inf")
    if "x" not in constraints: a[..., 2*d:3*d, :] = float("-inf")
    if "y" not in constraints: a[..., 3*d:, :] = float("-inf")

    return a

class PretrainedInitializer:

    def __init__(self, tensor: FloatTensor) -> None:
        self.tensor = tensor

    def __call__(self, x: FloatTensor) -> FloatTensor:
        return self.tensor.clone()

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

        # au = 0.25
        # av = 0.25
        # ax = 0.25
        # ay = 0.25

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
        r_pretrained: Sequence[Embedding] = None,
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

        if r_pretrained:
            for args, rr in zip(r_kwargs, r_pretrained):
                args["initializer"] = PretrainedInitializer(rr())

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
        s, u = r

        sx, sy = s.tensor_split(2, dim=-2)
        ux, uy = u.tensor_split(2, dim=-2)

        x = h.unsqueeze(-2)
        y = t.unsqueeze(-2)

        dist_x = sx * x + y - ux
        dist_y = sy * y + x - uy

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
        r_pretrained: Sequence[Embedding] = None,
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

        if r_pretrained:
            for args, rr in zip(r_kwargs, r_pretrained):
                args["initializer"] = PretrainedInitializer(rr())

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

class SWInteraction(Interaction):

    def __init__(
            self,
            *,
            scales = "x",
    ):
        super().__init__()

        self.entity_shape = ("d",)
        self.relation_shape = (("e", "d"), ("e", "d"), ("e", "d"))

        self.scales = scales

    def forward(self, h, r, t):
        s, w, u = r

        sx, sy = s.tensor_split(2, dim=-2)
        wx, wy = w.tensor_split(2, dim=-2)
        ux, uy = u.tensor_split(2, dim=-2)

        x = h.unsqueeze(-2)
        y = t.unsqueeze(-2)

        # dist_x = relu((sx * x + y - ux).abs() - wx)
        # dist_y = relu((sy * y + x - uy).abs() - wy)

        # # TODO zero-grad doesn't help while training...

        # dist_cat = []
        # if "x" in self.scales: dist_cat.append(dist_x)
        # if "y" in self.scales: dist_cat.append(dist_y)

        # dist = cat(dist_cat, dim=-1).squeeze(-2)

        # return (1 - vector_norm(dist, dim=-1)).sigmoid()

        dist_x = leaky_relu((sx * x + y - ux).abs() - wx)
        dist_y = leaky_relu((sy * y + x - uy).abs() - wy)

        dist_cat = []
        if "x" in self.scales: dist_cat.append(dist_x)
        if "y" in self.scales: dist_cat.append(dist_y)

        dist = cat(dist_cat, dim=-1).squeeze(-2)

        return (1 - dist.sum(dim=-1)).sigmoid()
    
    def filter(self, scales, a):
        if "x" not in scales: a[...,0,:] = float("-inf")
        if "y" not in scales: a[...,1,:] = float("-inf")

        return a

class SWModel(ERModel):

    def __init__(
        self,
        *,
        embedding_dim: int = 40,
        r_pretrained: Sequence[Embedding] = None,
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
                shape=(2, embedding_dim),
                initializer=xavier_uniform_
            ),
            dict( # width
                shape=(2, embedding_dim),
                initializer=xavier_uniform_
            ),
            dict( # offset
                shape=(2, embedding_dim),
                # FIXME init should be per component
                initializer=xavier_uniform_
            ),
        ]

        if r_pretrained:
            for args, rr in zip(r_kwargs, r_pretrained):
                args["initializer"] = PretrainedInitializer(rr())

        super().__init__(
            interaction=SWInteraction,
            interaction_kwargs=dict(
                scales=scales,
            ),
            entity_representations=Embedding,
            entity_representations_kwargs=e_kwargs,
            relation_representations=Embedding,
            relation_representations_kwargs=r_kwargs,
            **kwargs
        )

        self.name = scales + "w_scale"

class PolygonInteraction(Interaction):

    def __init__(
            self,
    ):
        super().__init__()

        self.entity_shape = ("d",)
        self.relation_shape = (("e", "d"), ("e", "d"))

    def forward(self, h, r, t):
        s, u = r

        sx, sy = s.tensor_split(2, dim=-2)
        ux, uy = u.tensor_split(2, dim=-2)

        x = h.unsqueeze(-2)
        y = t.unsqueeze(-2)

        # dist_x = relu(sx * x + y - ux)
        # dist_y = relu(sy * y + x - uy)

        dist_x = leaky_relu(sx * x + y - ux)
        dist_y = leaky_relu(sy * y + x - uy)

        # dist = cat((dist_x, dist_y), dim=-1).squeeze(-2)
        dist = cat((dist_x, dist_y), dim=-2).sum(dim=-2)

        return (1 - dist.sum(dim=-1)).sigmoid()

class PolygonModel(ERModel):
    
    def __init__(
        self,
        *,
        embedding_dim: int = 40,
        r_pretrained: Sequence[Embedding] = None,
        scales: int = 1,
        n: int = 2,
        constraints: str = "uvxy",
        edges: int = 2,
        **kwargs
    ) -> None:
        e_kwargs = dict(
            embedding_dim=embedding_dim,
            initializer=xavier_uniform_
        )

        r_kwargs = [
            dict( # scale
                shape=(2 * edges, embedding_dim),
                # initializer=uniform_,
                # initializer_kwargs=dict(a=-1, b=1)
                initializer=xavier_uniform_
            ),
            dict( # offset
                shape=(2 * edges, embedding_dim),
                initializer=xavier_uniform_
            ),
        ]

        if r_pretrained:
            for args, rr in zip(r_kwargs, r_pretrained):
                args["initializer"] = PretrainedInitializer(rr())
                args["initializer_kwargs"] = dict()

        super().__init__(
            interaction=PolygonInteraction,
            entity_representations=Embedding,
            entity_representations_kwargs=e_kwargs,
            relation_representations=Embedding,
            relation_representations_kwargs=r_kwargs,
            **kwargs
        )

        self.name = str(edges) + "_poly"

class FFNInteraction(Interaction):

    def __init__(
            self,
    ):
        super().__init__()

        self.entity_shape = ("d",)
        self.relation_shape = (("e", "h"), ("h", "f"), ("d",))

    def forward(self, h, r, t):
        w1, w2, b1 = r

        # if h.size(0) == 1:
        #     h = h.tile((t.size(0), 1, 1))
        #     t = t.tile((1, h.size(1), 1))
        # if t.size(0) == 1:
        #     t = t.tile((h.size(0), 1, 1))
        #     h = h.tile((1, t.size(1), 1))

        # ht = cat((h, t), dim=-1).unsqueeze(-2)
        
        # l1 = ht @ w1 + b1.unsqueeze(-2)

        ###

        h = h.unsqueeze(-2)
        t = t.unsqueeze(-2)

        wh1, wt1 = w1.tensor_split(2, dim=-2)
        # same as cat((h, t)) @ w1 + b1
        l1 = h @ wh1 + t @ wt1 + b1.unsqueeze(-2)

        ###

        score = where(l1 > 0, l1, 0) @ w2

        return score.squeeze(-2).sigmoid()

class FFNModel(ERModel):

    def __init__(
        self,
        *,
        embedding_dim: int = 40,
        r_pretrained: Sequence[Embedding] = None,
        scales: int = 1,
        n: int = 2,
        constraints: str = "uvxy",
        **kwargs
    ) -> None:
        e_kwargs = dict(
            embedding_dim=embedding_dim,
            initializer=xavier_uniform_
        )

        # TODO as arg
        hidden_dim = 40

        r_kwargs = [
            dict( # w1
                shape=(2 * embedding_dim, hidden_dim),
                initializer=xavier_uniform_
            ),
            dict( # w2
                shape=(hidden_dim, 1),
                initializer=xavier_uniform_
            ),
            dict( # b1
                shape=(hidden_dim),
                initializer=xavier_uniform_
            ),
        ]

        if r_pretrained:
            for args, rr in zip(r_kwargs, r_pretrained):
                args["initializer"] = PretrainedInitializer(rr())

        super().__init__(
            interaction=FFNInteraction,
            entity_representations=Embedding,
            entity_representations_kwargs=e_kwargs,
            relation_representations=Embedding,
            relation_representations_kwargs=r_kwargs,
            **kwargs
        )

        self.name = "ffn"