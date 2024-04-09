import argparse

def get_args() -> dict:
    parser = argparse.ArgumentParser()

    parser.add_argument("--cudaid", type=str, default=0,
                        help="cuda id: '0,1,2,3'")
    parser.add_argument("--fusion_technique", type=str, default=None,
                        help="Different transformer fusion techniques:"
                        "'Multi_cross_attention', 'Two_cross_attention',"
                        "'Feature concat'")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size")
    parser.add_argument("--lr_backbones", type=float, default=None,
                        help="Learning rate for backbones")
    parser.add_argument("--lr_fusion", type=float, default=None,
                        help="Learning rate for the fusion model")
    
    input_parser = parser.parse_args()

    return vars(input_parser)