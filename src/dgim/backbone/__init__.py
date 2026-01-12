from .backbone import RepVGG_8_2_align


def build_backbone(config):
    if config['backbone_type'] == 'RepVGG':
        if config['align_corner'] is False:
            if config['resolution'] == (8, 2):
                return RepVGG_8_2_align(config['backbone'])
        else:
            raise ValueError(f"DGIM.ALIGN_CORNER {config['align_corner']} not supported.")
    else:
        raise ValueError(f"DGIM.BACKBONE_TYPE {config['backbone_type']} not supported.")
