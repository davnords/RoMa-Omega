import random
import numpy as np

import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    logger.info(f"Setting seed to {seed}")
    import torch
    import cv2

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # cv2.findFundamentalMat(..., USAC_MAGSAC) draws from OpenCV's own RNG,
    # untouched by random/np.random/torch seeding above.
    cv2.setRNGSeed(seed)
