import argparse



def get_args():
	parser = argparse.ArgumentParser(description='FC and SC Classification')

	#* TRAINING parameters 
	parser.add_argument('--root', type=str)
	parser.add_argument('--path', type=str, default=None, help="path of the dataset")
	parser.add_argument('--result_path', type=str, default=None, help="path of the saved-results")
	parser.add_argument('--device', type=str, default='cpu')
	parser.add_argument('--seed', type=int, default=777, help='random seed')
	parser.add_argument('--dataset', type=str, default=None, help='name of the dataset')
	parser.add_argument('--folds', type=int, default=10, help='number of folds (default: 10)')
	parser.add_argument('--times', type=int, default=10, help='number of repetitions (default: 10)')
	parser.add_argument('--batch_size', type=int, default=128, help='batch size')
	parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
	parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
	parser.add_argument('--threshold', type=float, default=0.12, help='threshold')
	parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
	parser.add_argument('--patience', type=int, default=300, help='patience for early stopping')
	parser.add_argument('--mode', type=str, default='train', help="[train, KD]")
	parser.add_argument('--stage', type=str, default="contrast", help="[contrast, ce, mix]")
	parser.add_argument('--backbone', type=str, default=None, help="type of fusion backbone")
	parser.add_argument('--finetune_patience', type=int, default=300)

	#* MODEL parameters 
	parser.add_argument('--conv', type=str, default="GCN", help="type of GNNs, [GCN, GAT, SAGE, GIN]")
	parser.add_argument('--in_size', type=int, default=90)
	parser.add_argument('--num_classes', type=int, default=2, help='the number of classes (HC/MDD)')
	parser.add_argument('--hidden_dim', type=int, default=128, help='hidden size')
	parser.add_argument('--dropout', type=float, default=0.1, help='dropout ratio')
	parser.add_argument('--num_layers', type=int, default=4, help='the numbers of convolution layers')
	parser.add_argument('--pooling', type=str, default="add")

	#* FUSION MODEL parameters
	parser.add_argument('--fusion', type=str, default="MLP", help="MoE, MLP")
	parser.add_argument('--fusion_hidden', type=int, default=64)
	parser.add_argument('--num_fusion_layers', type=int, default=3)
	parser.add_argument('--alpha', type=float, default=0.6, help="hyper-parameter of loss disentanglement")
	parser.add_argument('--beta', type=float, default=0.3, help="hyper-parameter of loss KD")
	parser.add_argument('--num_experts', type=int, default=2, help="number of experiments within one MoE layer")
	parser.add_argument('--k', type=int, default=1, help="top-k token activatefor MoE. (1*token -> k*experts)")

	parser.add_argument('--num_heads', type=int, default=4, help="number of heads for transformer")
	parser.add_argument('--dim_ffn', type=int, default=128)
	parser.add_argument('--att_dropout', type=float, default=0.3)

	#* BASELINE
	parser.add_argument('--model_compare', type=str, default="GCN", help="TBD")
	parser.add_argument('--modal', type=str, default="fc", help="[sc, fc]")
	# parameter for Cross-GNN
	# parser.add_argument('--modal','-m1', type=int, default=0, help='0:fmri, 1:dti, 2:both')
	parser.add_argument('--kernel_size', type=int, default=90)
	parser.add_argument('--channel', type=int, default=32, help='channel number')  # 32
	parser.add_argument('--fold', type=int, default=-1, help='channel number')
	parser.add_argument('--layer', type=int, default=2, help='layer number')
	parser.add_argument('--ab', type=int, default=0, help='ablation study choice')
	parser.add_argument('--no', type=int, default=2, help='spit labels')
	parser.add_argument('--gru', type=int, default=1, help='layer number')

	args = parser.parse_args()
	return args
