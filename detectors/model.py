import skorch
import torch
# from detectors.dummy import DummyDetector
from detectors.retinanet import RetinaNet
from detectors.loss import DetectionLoss


def init(w):
    if w.dim() < 2:
        return w
    return torch.nn.init.xavier_normal_(w)


class DetectionNet(skorch.NeuralNet):
    def predict_proba(self, X):
        nonlin = self._get_predict_nonlinearity()
        y_probas = []
        for yp in self.forward_iter(X, training=True):
            for scale in nonlin(yp):
                y_probas.append(skorch.utils.to_numpy(scale))
        return y_probas

    def validation_step(self, batch, **fit_params):
        self._set_training(False)
        Xi, yi = skorch.dataset.unpack_data(batch)
        with torch.no_grad():
            y_pred = self.infer(Xi, **fit_params)
            losses = self.get_loss(y_pred, yi, X=Xi, training=False)
        losses["y_pred"] = y_pred
        return losses

    def train_step_single(self, batch, **fit_params):
        self._set_training(True)
        Xi, yi = skorch.dataset.unpack_data(batch)
        y_pred = self.infer(Xi, **fit_params)
        losses = self.get_loss(y_pred, yi, X=Xi, training=True)
        losses["loss"].backward()
        losses["y_pred"] = y_pred
        return losses

    def run_single_epoch(
            self, dataset, training, prefix, step_fn, **fit_params
    ):
        if dataset is None:
            return

        batch_count = 0
        for batch in self.get_iterator(dataset, training=training):
            self.notify("on_batch_begin", batch=batch, training=training)
            step = step_fn(batch, **fit_params)
            self.history.record_batch(prefix + "_loss", step["loss"].item())
            batch_size = (
                skorch.dataset.get_len(batch[0])
                if isinstance(batch, (tuple, list))
                else skorch.dataset.get_len(batch)
            )
            self.history.record_batch(prefix + "_batch_size", batch_size)
            self.notify("on_batch_end", batch=batch, training=training, **step)
            batch_count += 1

        self.history.record(prefix + "_batch_count", batch_count)


def build_model(max_epochs=2, logdir=".tmp/", train_split=None):
    # A slight improvement
    base_lr = 0.0002
    batch_size = 16

    # scheduler = skorch.callbacks.LRScheduler(
    #     policy=torch.optim.lr_scheduler.CyclicLR,
    #     base_lr=base_lr,
    #     max_lr=0.001,
    #     step_size_up=batch_size * 4,
    #     step_size_down=batch_size * 5,
    #     step_every='batch',
    #     mode="triangular2",
    # )

    model = DetectionNet(
        RetinaNet,
        batch_size=batch_size,
        max_epochs=max_epochs,
        lr=base_lr,
        criterion=DetectionLoss,
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
            # scheduler,
            skorch.callbacks.ProgressBar(),
            skorch.callbacks.TrainEndCheckpoint(dirname=logdir),
            skorch.callbacks.Initializer("*", init),
            skorch.callbacks.PassthroughScoring(
                name='boxes',
                on_train=True,
            ),
            skorch.callbacks.PassthroughScoring(
                name='train_classes',
                on_train=True,
            ),
        ],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    return model
