import logging
import os
import shutil
import tempfile

import torch
from pytorch_lightning.callbacks import BasePredictionWriter

logger = logging.getLogger(__name__)


class CustomWriter(BasePredictionWriter):
    """Pytorch Lightning Callback that saves predictions and the corresponding batch
    indices in a temporary folder when using multigpu inference.

    Args:
        write_interval (str): When to perform write operations. Defaults to 'epoch'
    """

    def __init__(self, write_interval="epoch") -> None:
        super().__init__(write_interval)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        """Saves predictions after running inference on all samples."""

        # We need to save predictions in the most secure manner possible to avoid
        # multiple users and processes writing to the same folder.
        # For that we will create a tmp folder that will be shared only across
        # the DDP processes that were created
        if trainer.is_global_zero:
            output_dir = [
                tempfile.mkdtemp(),
            ]
            logger.info(
                "Created temporary folder to store predictions: {}.".format(
                    output_dir[0]
                )
            )
        else:
            output_dir = [
                None,
            ]

        torch.distributed.broadcast_object_list(output_dir)

        # Make sure every process received the output_dir from RANK=0
        torch.distributed.barrier()  
        # Now that we have a single output_dir shared across processes we can save
        # prediction along with their indices.
        self.output_dir = output_dir[0]
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(
            predictions, os.path.join(self.output_dir, f"pred_{trainer.global_rank}.pt")
        )
        # optionally, you can also save `batch_indices` to get the information about
        # the data index from your prediction data
        torch.save(
            batch_indices,
            os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"),
        )

    def gather_all_predictions(self):
        """Reads all saves predictions from the self.output_dir into one single
        Prediciton object respecting the original order of the samples.
        """
        files = sorted(os.listdir(self.output_dir))
        predictions = [torch.load(os.path.join(self.output_dir, f))[0] for f in files if "pred" in f]
       
        return predictions

    def cleanup(self):
        """Cleans temporary files."""
        logger.info("Cleanup temporary folder: {}.".format(self.output_dir))
        shutil.rmtree(self.output_dir)