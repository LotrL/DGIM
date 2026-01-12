from yacs.config import CfgNode as CN


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


_CN = CN()
# _CN.BACKBONE_TYPE = 'ResNetFPN'
# _CN.RESOLUTION = (8, 2)  # options: [(8, 2), (16, 4)]
_CN.BACKBONE_TYPE = 'RepVGG'
_CN.ALIGN_CORNER = False
_CN.RESOLUTION = (8, 2)

_CN.FINE_WINDOW_SIZE = 5  # window_size in fine_level, must be odd
_CN.FINE_CONCAT_COARSE_FEAT = True

_CN.MP = False
_CN.HALF = False

# 1. DGIM-backbone (local feature CNN) config
_CN.BACKBONE = CN()
_CN.BACKBONE.BLOCK_DIMS = [128, 196, 256]  # s1, s2, s3

# 2. DGIM-coarse module config
_CN.COARSE = CN()
_CN.COARSE.D_MODEL = 256
_CN.COARSE.D_FFN = 256
_CN.COARSE.NHEAD = 8
_CN.COARSE.LAYER_NAMES = ['self', 'cross'] * 4  # default: ['self', 'conv', 'cross'] * 4
_CN.COARSE.ATTENTION = 'linear'  # options: ['linear', 'full']
_CN.COARSE.TEMP_BUG_FIX = False

_CN.COARSE.AGG_SIZE0 = 2  # default: 4
_CN.COARSE.AGG_SIZE1 = 2  # default: 4
_CN.COARSE.AGG_SIZE = 4  # default: 4
_CN.COARSE.NO_FLASH = False
_CN.COARSE.ROPE = True
_CN.COARSE.NPE = [832, 832, 832, 832]  # [832, 832, long_side, long_side] Suggest setting based on the long side of the input image, especially when the long_side > 832

# 3. Coarse-Matching config
_CN.MATCH_COARSE = CN()
_CN.MATCH_COARSE.THR = 0.3  # default: 0.3
_CN.MATCH_COARSE.BORDER_RM = 2
_CN.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'  # options: ['dual_softmax, 'sinkhorn']
_CN.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.MATCH_COARSE.SKH_ITERS = 3
_CN.MATCH_COARSE.SKH_INIT_BIN_SCORE = 1.0
_CN.MATCH_COARSE.SKH_PREFILTER = True
_CN.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.4  # training tricks: save GPU memory
_CN.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock

# 4. DGIM-fine module config
_CN.FINE = CN()
_CN.FINE.D_MODEL = 128
_CN.FINE.D_FFN = 128
_CN.FINE.NHEAD = 8
_CN.FINE.LAYER_NAMES = ['self', 'cross'] * 1
_CN.FINE.ATTENTION = 'linear'

_CN.FINE.DSMAX_TEMPERATURE = 0.1
_CN.FINE.THR = 0.1

default_cfg = lower_config(_CN)
