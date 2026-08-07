import torch
import logging

logger = logging.getLogger(__name__)
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    logger.warning("MPS is untested, use with caution.")
    device = torch.device("mps")
else:
    device = torch.device("cpu")
