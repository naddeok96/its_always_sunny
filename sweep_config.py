sweep_config = {
                'method': 'grid',
                'metric': {
                'name': 'loss',
                'goal': 'minimize'   
                },
                'parameters': {
                    'epochs': {
                        'values': [1e5]
                    },
                    'batch_size': {
                        'values': [1]
                    },
                    'n_nodes_fc1' : {
                        'values': [512]
                    },
                    'n_nodes_fc2' : {
                        'values'  : [256]
                    },
                    'embedding_size': {
                        'values': [2]
                    },
                    'momentum': {
                        'values': [0.9]
                    },
                    'weight_decay': {
                        'values': [0.005]
                    },
                    'learning_rate': {
                        'values': [1e-3]
                    },
                    'optimizer': {
                        'values': ['sgd']
                    },
                    'criterion': {
                        'values': ["mse"]
                    },
                    'activation_func': {
                        'values': ['relu']
                    },
                    'output_activation_func': {
                        'values': ["softmax"]
                    }
                    
                }
            }