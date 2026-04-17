from collections.abc import Sequence

from torch import tensor, prod, cat, where, FloatTensor
from torch.linalg import vector_norm
from torch.nn.functional import relu, gelu, leaky_relu, softplus
from torch.nn.init import xavier_uniform_, uniform_, normal_

from pykeen.nn import Embedding
from pykeen.nn.modules import Interaction
from pykeen.nn.representation import Representation
from pykeen.models import ERModel

class FixedValueInitializer:

    def __init__(self, values: Sequence[int]) -> None:
        self.values = values

    def __call__(self, x: FloatTensor) -> FloatTensor:
        for i, v in enumerate(self.values): x[..., i, :] = v

        return x

class PretrainedInitializer:

    def __init__(self, tensor: FloatTensor) -> None:
        self.tensor = tensor

    def __call__(self, x: FloatTensor) -> FloatTensor:
        return self.tensor.clone()

class EdgeInteraction(Interaction):

    def __init__(
        self,
        *,
        margin = 1,
        symmetric = False,
    ):
        super().__init__()

        self.margin = margin
        self.symmetric = symmetric

        self.entity_shape = ("d",)
        self.relation_shape = (("e", "d"), ("e", "d"), ("e", "d"), ("e", "d"))

    def forward(self, h, r, t):
        s, u, w, a = r

        a = a.softmax(dim=-2)

        w = w.abs()

        x = h.unsqueeze(-2)
        y = t.unsqueeze(-2)

        if a.size(-2) >= 4:
            s1, s2, s3, s4 = s.tensor_split(4, dim=-2)
            u1, u2, u3, u4 = u.tensor_split(4, dim=-2)
            w1, w2, w3, w4 = w.tensor_split(4, dim=-2)
            a1, a2, a3, a4 = a.tensor_split(4, dim=-2)

            dist_1 = self.f(a1, w1, s1 * x - y - u1)
            dist_2 = self.f(a2, w2, s2 * y - x - u2)
            dist_3 = self.f(a3, w3, s3 * x + y - u3)
            dist_4 = self.f(a4, w4, s4 * y + x - u4)

            # TODO reshape instead of summing?
            dist = cat((dist_1, dist_2, dist_3, dist_4), dim=-2).sum(dim=-2)
        elif a.size(-2) >= 2:
            sx, sy = s.tensor_split(2, dim=-2)
            ux, uy = u.tensor_split(2, dim=-2)
            wx, wy = w.tensor_split(2, dim=-2)
            ax, ay = a.tensor_split(2, dim=-2)

            dist_x = self.f(ax, wx, sx * x + y - ux)
            dist_y = self.f(ay, wy, sy * y + x - uy)

            dist = cat((dist_x, dist_y), dim=-2).sum(dim=-2)
        else:
            dist = self.f(a, w, s * x + y - u).squeeze(-2)

        # return (1 - dist.sum(dim=-1)).sigmoid()
        return (self.margin - vector_norm(dist, dim=-1)).sigmoid()

    def f(self, a, w, dist):
        if self.symmetric: dist = dist.abs()
        return a * (dist - w).relu()

class RegionBasedModel(ERModel):

    def __init__(
        self,
        *,
        r_pretrained: Sequence[Embedding] = None,
        embedding_dim: int = 40,
        margin: float = 1,
        edges: int = 2,
        symmetric: bool = True, # relu(cat(dist, -dist)) ~ abs
        scales: Sequence[float] = None, # octagon = [-1, 1, 0, 0]
        widths: Sequence[float] = None,
        attention_scores: Sequence[float] = None,
        **kwargs
    ) -> None:
        e_kwargs = dict(
            embedding_dim=embedding_dim,
            initializer=xavier_uniform_
        )

        r_kwargs = [
            dict( # s
                shape=(edges, embedding_dim),
                initializer=xavier_uniform_
            ),
            dict( # u
                shape=(edges, embedding_dim),
                initializer=xavier_uniform_
            ),
            dict( # w
                shape=(edges, embedding_dim)
            ),
            dict( # a
                shape=(edges, embedding_dim)
            ),
        ]

        if scales is not None:
            scale_kwargs = r_kwargs[0]

            scale_kwargs["trainable"] = False
            scale_kwargs["initializer"] = FixedValueInitializer(scales)

        if widths is not None:
            width_kwargs = r_kwargs[2]

            width_kwargs["trainable"] = False
            width_kwargs["initializer"] = FixedValueInitializer(widths)

        if attention_scores is not None:
            a_kwargs = r_kwargs[3]

            a_kwargs["trainable"] = False
            a_kwargs["initializer"] = FixedValueInitializer(attention_scores)

        if r_pretrained:
            for args, rr in zip(r_kwargs, r_pretrained):
                args["initializer"] = PretrainedInitializer(rr())

        super().__init__(
            interaction=EdgeInteraction,
            interaction_kwargs=dict(
                margin=margin,
                symmetric=symmetric
            ),
            entity_representations=Embedding,
            entity_representations_kwargs=e_kwargs,
            relation_representations=Embedding,
            relation_representations_kwargs=r_kwargs,
            **kwargs
        )

        sym_opt = "_sym" if symmetric else ""
        scales_opt = "_s" if scales else ""
        # TODO add widths_opt
        self.name = f"{edges}{sym_opt}{scales_opt}_model"