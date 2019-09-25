import argparse

def parseOpts():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ct-dir',
                        default='data/dataset/EGFR/egfr_prep_result')

    parser.add_argument('--bbox-csv',
                        default='data/dataset/EGFR/predicted_bbox.csv')

    parser.add_argument('--train-csv',
                        default='data/dataset/EGFR/train_simple.csv')

    parser.add_argument('--val-csv',
                        default='data/dataset/EGFR/val_simple.csv')

    parser.add_argument('--batch-size',
                        default=32,
                        type=int)

    parser.add_argument('--lr',
                        default=0.001,
                        type=float)

    parser.add_argument('--log-dir',
                        default='data/log')

    parser.add_argument('--weight',
                        default='medicalnet_resnet50.pth')

    parser.add_argument('--gpus',
                        default='0')

    args = parser.parse_args()
    
    return args
