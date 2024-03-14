from argparse import ArgumentParser, Namespace


def add_train_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    # General arguments
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--split_path', type=str)
    parser.add_argument('--trained_local_model', type=str)
    parser.add_argument('--restart_dir', type=str)
    parser.add_argument('--dataset', type=str, default='qm9')
    parser.add_argument('--seed', type=int, default=0)

    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--scheduler', type=str, default='plateau')
    parser.add_argument('--verbose', action='store_true', default=False)

    # Model arguments
    parser.add_argument('--model_dim', type=int, default=25)
    parser.add_argument('--random_vec_dim', type=int, default=10)
    parser.add_argument('--random_vec_std', type=float, default=1)
    parser.add_argument('--random_alpha', action='store_true', default=False)
    parser.add_argument('--n_true_confs', type=int, default=10)
    parser.add_argument('--n_model_confs', type=int, default=10)

    parser.add_argument('--gnn1_depth', type=int, default=3)
    parser.add_argument('--gnn1_n_layers', type=int, default=2)
    parser.add_argument('--gnn2_depth', type=int, default=3)
    parser.add_argument('--gnn2_n_layers', type=int, default=2)
    parser.add_argument('--encoder_n_head', type=int, default=2)
    parser.add_argument('--coord_pred_n_layers', type=int, default=2)
    parser.add_argument('--d_mlp_n_layers', type=int, default=1)
    parser.add_argument('--h_mol_mlp_n_layers', type=int, default=1)
    parser.add_argument('--alpha_mlp_n_layers', type=int, default=2)
    parser.add_argument('--c_mlp_n_layers', type=int, default=1)

    parser.add_argument('--global_transformer', action='store_true', default=False)
    parser.add_argument('--loss_type', type=str, default='ot_emd')
    parser.add_argument('--teacher_force', action='store_true', default=False)
    parser.add_argument('--separate_opts', action='store_true', default=False)


def parse_train_args() -> Namespace:
    """
    Parses arguments for training (includes modifying/validating arguments).

    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()

    return args


def set_hyperparams(args):
    """
    Converts ArgumentParser args to hyperparam dictionary.

    :param args: Namespace containing the args.
    :return: A dictionary containing the args as hyperparams.
    """

    hyperparams = {'model_dim': args.model_dim,
                   'random_vec_dim': args.random_vec_dim,
                   'random_vec_std': args.random_vec_std,
                   'global_transformer': args.global_transformer,
                   'n_true_confs': args.n_true_confs,
                   'n_model_confs': args.n_model_confs,
                   'gnn1': {'depth': args.gnn1_depth,
                            'n_layers': args.gnn1_n_layers},
                   'gnn2': {'depth': args.gnn2_depth,
                            'n_layers': args.gnn2_n_layers},
                   'encoder': {'n_head': args.encoder_n_head},
                   'coord_pred': {'n_layers': args.coord_pred_n_layers},
                   'd_mlp': {'n_layers': args.d_mlp_n_layers},
                   'h_mol_mlp': {'n_layers': args.h_mol_mlp_n_layers},
                   'alpha_mlp': {'n_layers': args.alpha_mlp_n_layers},
                   'c_mlp': {'n_layers': args.c_mlp_n_layers},
                   'loss_type': args.loss_type,
                   'teacher_force': args.teacher_force,
                   'random_alpha': args.random_alpha}

    return hyperparams
