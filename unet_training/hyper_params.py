import argparse

hp_parser = argparse.ArgumentParser(description='training')
# settings
hp_parser.add_argument('--gpu_id', type=int, help='gpu id to use')
hp_parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')

# hyperparameters
hp_parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
hp_parser.add_argument('--batch_size', type=int, default=16)
hp_parser.add_argument('--infer_batch_size', type=int, default=16, help='batch size during validation and testing')
hp_parser.add_argument('--weight_decay', type=float, default=0, help='weight decay used by optimizer')
hp_parser.add_argument('--epochs', type=int, default=350, help='number of epochs for training')

# architecture
hp_parser.add_argument('--n_last_channel', type=int, default=64, help='number of channels before the last convolution')
