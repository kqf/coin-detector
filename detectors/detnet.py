import skorch
import matplotlib.pyplot as plt
import torch

from detectors.shapes import box


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

            for name, output in step.items():
                if name == "y_pred":
                    continue
                self.history.record_batch(f"{prefix}_{name}", output.item())

            batch_size = (
                skorch.dataset.get_len(batch[0])
                if isinstance(batch, (tuple, list))
                else skorch.dataset.get_len(batch)
            )
            self.history.record_batch(prefix + "_batch_size", batch_size)
            self.notify("on_batch_end", batch=batch, training=training, **step)
            batch_count += 1

        self.history.record(prefix + "_batch_count", batch_count)


class DebugDetectionNet(DetectionNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        y_true = skorch.utils.to_tensor(y_true, device=self.device)

        class Debug:
            def __init__(self, name, subloss, images):
                self.name = name
                self.subloss = subloss
                self.needs_negatives = subloss.needs_negatives
                self.images = images

            def __call__(self, y_pred, y_true, anchors):
                for i, image in enumerate(self.images):
                    channels_last = image.permute(2, 1, 0).numpy()
                    plt.imshow(channels_last)

                    if not self.needs_negatives:
                        for coords in y_true:
                            box(channels_last, *coords)

                    for coords in anchors:
                        box(channels_last, *coords, color="r")

                    plt.savefig(f"{self.name}-{i}.png")
                    plt.show()
                return self.subloss(y_pred, y_true, anchors)

        sublosses = self.criterion_.sublosses
        deblosses = {name: Debug(name, l, X) for name, l in sublosses.items()}
        self.criterion_.sublosses = deblosses
        loss = self.criterion_(y_pred, y_true)
        self.criterion_.sublosses = sublosses
        return loss
