"""Native SimATTA supplies the shared D=0 query/update protocol."""

from collections import Counter

from experiments.fcra_ablation.algorithm import OriginalSimATTA, AblationFCRA


class NativeSimATTA(OriginalSimATTA):
    def __init__(self, config):
        super().__init__(config)
        self.stream_step = -1
        self.iterations = Counter()

    def select_queries(self, data, stream_ids):
        self.stream_step += 1
        return super().select_queries(data, stream_ids)

    def one_step_train(self, *args, **kwargs):
        loss = super().one_step_train(*args, **kwargs)
        self.update_rows[-1].update(step=self.stream_step,
                                   iteration=self.iterations[self.stream_step])
        self.iterations[self.stream_step] += 1
        return loss

    def run_stream(self, output_dir, progress=None):
        result = super().run_stream(output_dir, progress=progress)
        result.update(method="VLCS-calibration", variant="simatta_original", dataset="VLCS")
        return result


class ControlledCalibration(AblationFCRA):
    def run_stream(self, output_dir, progress=None):
        result = super().run_stream(output_dir, progress=progress)
        result.update(method="VLCS-calibration", dataset="VLCS")
        return result
