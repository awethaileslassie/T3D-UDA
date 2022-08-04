from nni.experiment import Experiment

# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(parent_dir)

# params = {'batch_size': 32,'hidden_size1': 128,'hidden_size2': 128, 'lr': 0.001,'momentum': 0.5}

search_space = {
    'lr': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
    'momentum': {'_type': 'uniform', '_value': [0, 1]},
}
experiment = Experiment('local')

experiment.config.trial_code_directory = '.'
experiment.config.trial_command = "../script/sbatch train_wod_f1_1.sh"
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.max_trial_number = 10
experiment.config.trial_gpu_number = 0
experiment.config.trial_concurrency = 2
experiment.run(8080)

input('Press enter to quit')
experiment.stop()
