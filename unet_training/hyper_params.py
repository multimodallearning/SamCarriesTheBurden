import argparse

hp_parser = argparse.ArgumentParser(description='training')
# settings
hp_parser.add_argument('--gpu_id', type=int, help='gpu id to use')
hp_parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')

# hyperparameters
hp_parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
hp_parser.add_argument('--batch_size', type=int, default=8)
hp_parser.add_argument('--infer_batch_size', type=int, default=16, help='batch size during validation and testing')
hp_parser.add_argument('--weight_decay', type=float, default=0, help='weight decay used by optimizer')
hp_parser.add_argument('--epochs', type=int, default=500, help='number of epochs for training')

# lr scheduler: reduce on plateau
hp_parser.add_argument('--lr_patience', type=int, default=400, help='number of epochs to wait before reducing lr')
hp_parser.add_argument('--lr_gamma', type=float, default=0.98, help='factor to reduce lr')
hp_parser.add_argument('--lr_scheduler', default=True, action=argparse.BooleanOptionalAction,
                       help='whether to use lr scheduler')

# architecture
hp_parser.add_argument('--n_last_channel', type=int, default=128, help='number of channels before the last convolution')