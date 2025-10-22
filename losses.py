from pykeen.losses import AdversarialLoss, PointwiseLoss, apply_label_smoothing
from torch import FloatTensor, ones_like, zeros_like, max
from torch.nn.functional import binary_cross_entropy

class AdversarialHingeLoss(AdversarialLoss):

    def positive_loss_term(
        self,
        pos_scores: FloatTensor,
        label_smoothing: float = None,
        num_entities: int = None
    ) -> FloatTensor:
        input = pos_scores
        target = apply_label_smoothing(ones_like(pos_scores), epsilon=label_smoothing, num_classes=num_entities)

        return (-input * target).mean()
    
    def negative_loss_term_unreduced(
        self,
        neg_scores: FloatTensor,
        label_smoothing: float = None,
        num_entities: int = None
    ) -> FloatTensor:
        input = neg_scores
        target = apply_label_smoothing(ones_like(neg_scores), epsilon=label_smoothing, num_classes=num_entities)
        
        return input * target

class AdversarialBCEWithoutSigmoid(AdversarialLoss):

    def positive_loss_term(
        self,
        pos_scores: FloatTensor,
        label_smoothing: float = None,
        num_entities: int = None
    ) -> FloatTensor:
        return binary_cross_entropy(
            pos_scores,
            # TODO: maybe we can make this more efficient?
            apply_label_smoothing(ones_like(pos_scores), epsilon=label_smoothing, num_classes=num_entities),
            reduction=self.reduction,
        )
    
    def negative_loss_term_unreduced(
        self,
        neg_scores: FloatTensor,
        label_smoothing: float = None,
        num_entities: int = None
    ) -> FloatTensor:
        return binary_cross_entropy(
            neg_scores,
            # TODO: maybe we can make this more efficient?
            apply_label_smoothing(zeros_like(neg_scores), epsilon=label_smoothing, num_classes=num_entities),
            reduction="none",
        )
    
class BCEWithoutSigmoid(PointwiseLoss):

    def forward(
        self,
        x: FloatTensor,
        target: FloatTensor,
        weight: FloatTensor | None = None
    ) -> FloatTensor:
        return binary_cross_entropy(
            x,
            target,
            reduction=self.reduction,
            weight=weight
        )