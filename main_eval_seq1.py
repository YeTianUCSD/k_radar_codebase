'''
Evaluate a trained RTNH sequence-1 checkpoint.
'''

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate RTNH on local K-Radar sequence 1')
    parser.add_argument('--config', default='./configs/cfg_RTNH_wide_seq1.yml')
    parser.add_argument('--model', required=True)
    parser.add_argument('--conf_thr', type=float, nargs='+', default=[0.3])
    parser.add_argument('--subset', action='store_true')
    args = parser.parse_args()

    pline = PipelineDetection_v1_0(args.config, mode='test')
    pline.load_dict_model(args.model)
    pline.network.eval()
    pline.validate_kitti_conditional(
        list_conf_thr=args.conf_thr,
        is_subset=args.subset,
        is_print_memory=False,
    )
