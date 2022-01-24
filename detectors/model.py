import skorch
import torch

# from detectors.dummy import DummyDetector
from detectors.loss import DetectionLoss, default_losses
from detectors.retinanet import RetinaNet
from detectors.detnet import DetectionNet


def init(w):
    if w.dim() < 2:
        return w
    return torch.nn.init.xavier_normal_(w)


def build_model(max_epochs=2, logdir=".tmp/", train_split=None):
    # A slight improvement
    base_lr = 0.0002
    batch_size = 4

    scheduler = skorch.callbacks.LRScheduler(
        policy=torch.optim.lr_scheduler.CyclicLR,
        base_lr=base_lr,
        max_lr=0.001,
        step_size_up=1,
        step_size_down=1,
        step_every='epoch',
        mode="triangular2",
    )

    sublosses = default_losses()
    sublosses["boxes"].weight = 0.05

    model = DetectionNet(
        RetinaNet,
        batch_size=batch_size,
        max_epochs=max_epochs,
        lr=base_lr,
        criterion=DetectionLoss,
        criterion__sublosses=sublosses,
        optimizer=torch.optim.Adam,
        # optimizer__momentum=0.9,
        iterator_train__shuffle=True,
        iterator_train__num_workers=6,
        iterator_valid__shuffle=False,
        iterator_valid__num_workers=6,
        train_split=train_split,
        # predict_nonlinearity=partial(
        #     infer,
        #     top_n=top_n,
        #     min_iou=0.5,
        #     threshold=0.5,
        # ),
        callbacks=[
            scheduler,
            skorch.callbacks.ProgressBar(),
            skorch.callbacks.TrainEndCheckpoint(dirname=logdir),
            skorch.callbacks.Initializer("*", init),
            skorch.callbacks.PassthroughScoring(
                name='train_boxes',
                on_train=True,
            ),
            skorch.callbacks.PassthroughScoring(
                name='train_classes',
                on_train=True,
            ),
            skorch.callbacks.PassthroughScoring(
                name='train_classes_f1',
                on_train=True,
                lower_is_better=False,
            ),
        ],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    return model
