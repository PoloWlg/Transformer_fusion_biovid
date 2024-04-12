import argparse
import os
import yaml


def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def outfd(exp_id):
    tag = 'id_' + exp_id

    parent_lv = "exps"
    subpath = os.path.join(parent_lv, tag)
    outd = os.path.join(os.getcwd(), subpath)
    os.makedirs(outd, exist_ok=True)
    os.makedirs(os.path.join(outd, 'weights'), exist_ok=True)

    return outd

def get_args() -> dict:
    parser = argparse.ArgumentParser()

    parser.add_argument("--cudaid", type=str, default=0,
                        help="cuda id: '0,1,2,3'")
    parser.add_argument("--fusion_technique", type=str, default=None,
                        help="Different transformer fusion techniques:"
                        "'Multi_cross_attention', 'Two_cross_attention',"
                        "'Feature_concat'")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size")
    parser.add_argument("--lr_backbones", type=float, default=None,
                        help="Learning rate for backbones")
    parser.add_argument("--lr_fusion", type=float, default=None,
                        help="Learning rate for the fusion model")
    parser.add_argument("--SEED", type=str2bool, default=None,
                        help="setting seed (Bool)")
    parser.add_argument("--exp_id", type=str, default=None,
                        help="Experiment Id")
    
    input_parser = parser.parse_args()

    args = vars(input_parser)

    args['outd'] = outfd(args['exp_id'])

    with open(os.path.join(args['outd'], "config.yml"), 'w') as fyaml:
        yaml.dump(args, fyaml)

    return args