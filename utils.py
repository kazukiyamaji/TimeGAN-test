

from xml.etree.ElementInclude import default_loader


def parse(parser):
        
        #データセット関係
        parser.add_argument(
                '--dataset_type',
                default="normal_sine",
                type=str)
        parser.add_argument(
                '--train_batch_size',
                default=4,
                type=int)
        parser.add_argument(
                '--test_batch_size',
                default=4,
                type=int)
        parser.add_argument(
                '--train_rate',
                default=0.7,
                type=float)
        parser.add_argument(
                '--num_data',
                default=1000,
                type=int)
        parser.add_argument(
                '--seq_len',
                default=20,
                type=int)
        parser.add_argument(
                '--sine_dim',
                default=10,
                type=int)
        parser.add_argument(
                '--padding_value',
                default=-2,
                type=int)

        #モデル関係
        parser.add_argument(
                '--hidden_dim',
                default=100,
                type=int)
        parser.add_argument(
                "--z_dim",
                default=100,
                type=int
        )
        parser.add_argument(
                "--num_layers",
                default=3,
                type=int
        )
        parser.add_argument(
                "--device",
                default="cpu",
                type=str
        )

        #最適化関係
        parser.add_argument(
                "--lr",
                default=1e-3,
                type=float
        )
        parser.add_argument(
                "--gamma",
                default=1,
                type=float
        )
        parser.add_argument(
                "--e_epochs",
                default=600,
                type=int  
        )
        parser.add_argument(
                "--g_epochs",
                default=600,
                type=int  
        )
        parser.add_argument(
                "--s_epochs",
                default=600,
                type=int  
        )
        parser.add_argument(
                "--dis_check",
                default=0.15,
                type=float
        )


        return parser