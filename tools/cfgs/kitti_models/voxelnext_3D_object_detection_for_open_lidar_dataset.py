CLASS_NAMES: ['car', 'pedestrian', 'motorcycle' , 'truck' , 'bus' , 'bicycle' , 'other_vehicles' ]

DATA_CONFIG:
    _BASE_CONFIG_: cfgs/dataset_configs/kitti_dataset.yaml
    DATA_AUGMENTOR:
        DISABLE_AUG_LIST: ['placeholder']
        AUG_CONFIG_LIST:

            - NAME: random_world_flip
              ALONG_AXIS_LIST: ['x']

            - NAME: random_world_rotation
              WORLD_ROT_ANGLE: [-0.78539816, 0.78539816]

            - NAME: random_world_scaling
              WORLD_SCALE_RANGE: [0.95, 1.05]

    DATA_PROCESSOR:
        - NAME: mask_points_and_boxes_outside_range
          REMOVE_OUTSIDE_BOXES: True

        - NAME: shuffle_points
          SHUFFLE_ENABLED: {
            'train': True,
            'test': True
          }

        - NAME: transform_points_to_voxels
          VOXEL_SIZE: [0.075, 0.075, 0.2]
          MAX_POINTS_PER_VOXEL: 10
          MAX_NUMBER_OF_VOXELS: {
            'train': 120000,
            'test': 160000
          }

POINT_FEATURE_ENCODING: {
    encoding_type: absolute_coordinates_encoding,
    used_feature_list: ['x', 'y', 'z', 'intensity'],
    src_feature_list: ['x', 'y', 'z', 'intensity'],
}

MODEL:
    NAME: CenterPoint

    VFE:
        NAME: MeanVFE

    BACKBONE_3D:
        NAME: VoxelResBackBone8x

    MAP_TO_BEV:
        NAME: HeightCompression
        NUM_BEV_FEATURES: 128 #256

    BACKBONE_2D:
        NAME: BaseBEVBackbone

        LAYER_NUMS: [5, 5]
        LAYER_STRIDES: [1, 2]
        NUM_FILTERS: [256,256] #[128, 256]
        UPSAMPLE_STRIDES: [1, 2]
        NUM_UPSAMPLE_FILTERS: [256, 256]

    DENSE_HEAD:
        NAME: CenterHead
        CLASS_AGNOSTIC: False

        CLASS_NAMES_EACH_HEAD: [
            ['car'], 
            ['truck'], #['truck', 'construction_vehicle'],
            ['bus'], #['bus', 'trailer'],
            ['motorcycle'], ##['barrier'],
            ['bicycle'],
            ['pedestrian'],
            ['other_vehicles']
        ]

        SHARED_CONV_CHANNEL: 64
        USE_BIAS_BEFORE_NORM: True
        NUM_HM_CONV: 2
        SEPARATE_HEAD_CFG:
            HEAD_ORDER: ['center', 'center_z', 'dim', 'rot' ] #['center', 'center_z', 'dim', 'rot', 'vel']
            HEAD_DICT: {
                'center': {'out_channels': 2, 'num_conv': 2},
                'center_z': {'out_channels': 1, 'num_conv': 2},
                'dim': {'out_channels': 3, 'num_conv': 2},
                'rot': {'out_channels': 2, 'num_conv': 2},
                #'vel': {'out_channels': 2, 'num_conv': 2},
            }

        TARGET_ASSIGNER_CONFIG:
            FEATURE_MAP_STRIDE: 8
            NUM_MAX_OBJS: 500
            GAUSSIAN_OVERLAP: 0.1
            MIN_RADIUS: 2

        LOSS_CONFIG:
            LOSS_WEIGHTS: {
                'cls_weight': 1.0,
                'loc_weight': 0.25,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2 ] #, 1.0, 1.0]
            }

        POST_PROCESSING:
            SCORE_THRESH: 0.1
            POST_CENTER_LIMIT_RANGE: [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
            MAX_OBJ_PER_SAMPLE: 500
            NMS_CONFIG:
                NMS_TYPE: nms_gpu
                NMS_THRESH: 0.2
                NMS_PRE_MAXSIZE: 1000
                NMS_POST_MAXSIZE: 83

    POST_PROCESSING:
        RECALL_THRESH_LIST: [0.3, 0.5, 0.7]

        EVAL_METRIC: kitti



OPTIMIZATION:
    BATCH_SIZE_PER_GPU: 4
    NUM_EPOCHS: 1000 #20

    OPTIMIZER: adam_onecycle
    LR: 0.001
    WEIGHT_DECAY: 0.01
    MOMENTUM: 0.9

    MOMS: [0.95, 0.85]
    PCT_START: 0.4
    DIV_FACTOR: 10
    DECAY_STEP_LIST: [35, 45]
    LR_DECAY: 0.1
    LR_CLIP: 0.0000001

    LR_WARMUP: False
    WARMUP_EPOCH: 1

    GRAD_NORM_CLIP: 10
