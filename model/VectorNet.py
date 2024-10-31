from torch import nn
from torchvision.models import swin_t


class SwinT_Match(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.backbone = swin_t(pretrained=True)
        self.backbone.head = nn.Identity()
        self.fc = nn.Linear(768, 17)

    def forward(self, template, moving=None):
        if len(template.size()) == 5:
            template = template.squeeze(0)
            moving = moving.squeeze(0)
        embedding_template = self.backbone(template)
        embedding_moving = self.backbone(moving)
        embedding_template_cls = embedding_template
        embedding_moving_cls = embedding_moving

        classes_t = self.fc(embedding_template_cls)
        classes_m = self.fc(embedding_moving_cls)

        return (embedding_template_cls, embedding_moving_cls), (classes_t, classes_m)