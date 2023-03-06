def add_arguments(parser):
    '''
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''

    parser.add_argument('--video_dir', default = None, help = 'output video directory')
    parser.add_argument('--do_render', action = 'store_true', help = 'whether render environment')

    parser.add_argument('--batch_size', type = int, default = 64, help = 'batch size for training')
    parser.add_argument('--run_name', type = str, default = 'dqn_model', help = '')
    parser.add_argument('--model_save_path', type = str, default = 'models', help = '')
    parser.add_argument('--model_save_interval', type = int, default = 500, help = '')
    parser.add_argument('--log_path', type = str, default = 'train_log.out', help = '')
    parser.add_argument('--tensorboard_summary_path', type = str, default = 'tensorboard_summary', help = '')
    parser.add_argument('--model_test_path', type = str, default = 'models/dqn_model_69000.pt', help = '')
    parser.add_argument('--metrics_capture_window', type = int, default = 100, help = '')
    parser.add_argument('--replay_buffer_size', type = int, default = 10000, help = '')
    parser.add_argument('--start_to_learn', type = int, default = 5000, help = 'Initial number of entries in the buffer before learning starts')
    parser.add_argument('--total_num_steps', type = int, default = 5e7, help = '')
    parser.add_argument('--learning_rate', type = float, default = 1e-8, help= 'learning rate for the model')
    parser.add_argument('--gamma', type = float, default = 0.99, help = '')
    parser.add_argument('--initial_epsilon', type = float, default = 1.0, help = '')
    parser.add_argument('--final_epsilon', type = float, default = 0.005, help = '')
    parser.add_argument('--steps_to_explore', type=int, default = 1000000, help='')
    parser.add_argument('--network_update_interval', type = int, default = 5000, help = '')
    parser.add_argument('--episodes', type = int, default = 100000, help = '')
    parser.add_argument('--network_train_interval', type = int, default = 10, help = '')
    parser.add_argument('--ddqn', type = bool, default = True, help = '')
    
    return parser
