def add_dqn_args(parser):
    parser.add_argument('--name', type=str, default='dqn')
    parser.add_argument('--algo_type', type=str, default='rl')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--target_update', type=float, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--buffer_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_episode', type=int, default=1000)
    parser.add_argument('--minimal_size', type=int, default=500)
    parser.add_argument('--evaluate_freq', type=int, default=10)
    return parser