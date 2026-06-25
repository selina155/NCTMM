import os
import numpy as np


def get_dataset_path(dataset):
    if 'netsim' in dataset:
        dataset_path = 'netsim'
    elif 'dream3_combined' in dataset:
        dataset_path = 'dream3'
    elif 'yeast' in dataset:
        dataset_path = 'dream3'
    elif 'ecoli' in dataset:
        dataset_path = 'dream3'
    elif 'lorenz96' in dataset:
        dataset_path='lorenz96'
        
    elif dataset in ['traffic', 'medical', 'pm25', 'traffic_medical',"causaltime_cluster_0","causaltime_cluster_1"]:
        dataset_path = 'causaltime'
    elif 'aqi' in dataset:
        dataset_path='AQI'
    else:
        dataset_path = 'synthetic'

    return dataset_path


def create_save_name(dataset, cfg):
    if dataset == 'lorenz96':
        return f'lorenz96_N={cfg.num_nodes}_T={cfg.timesteps}_num_graphs={cfg.num_graphs}'
    return dataset


def norm(data):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    normalized_data = (data - mean) / (std)
    return normalized_data


def prepross_data(data):
    num_samples, T, num_nodes = data.shape
    new_data = np.zeros_like(data, dtype=float)
    for i in range(num_nodes):
        node = data[:, :, i]
        new_data[:, :, i] = node - node.mean()
    return new_data

def load_synthetic_from_folder(dataset_dir, dataset_file):
    #X = np.load(os.path.join(dataset_dir, dataset_file, 'grouped_by_graph/X_4.npy'))
    indices_to_remove=[4]
    X = np.load(os.path.join(dataset_dir, dataset_file, 'X.npy'))
    X = np.delete(X, indices_to_remove, axis=-1)
    #data = np.load(os.path.join(dataset_dir, dataset_file + '.npz'))
    #X=data["X"]
    #X=norm(X)
    #adj_matrix=data["adj_matrix"]
    adj_matrix=np.load(os.path.join(dataset_dir, dataset_file, 'adj_matrix.npy'))
    
    #X=X[:500]
    #adj_matrix = np.load(os.path.join(dataset_dir, dataset_file, 'grouped_by_graph/adj_matrix_4.npy'))
    #adj_matrix = np.load(os.path.join(dataset_dir, dataset_name, 'adj_matrix.npy'))
    adj_matrix=np.max(adj_matrix,axis=1)
    #print(adj_matrix.shape)
    #adj_matrix=np.expand_dims(adj_matrix,axis=0)
    #adj_matrix=np.tile(adj_matrix, (X.shape[0],1, 1))
    adj_matrix = np.delete(adj_matrix, indices_to_remove, axis=1)
    adj_matrix = np.delete(adj_matrix, indices_to_remove, axis=2)
    #print(adj_matrix.shape)
    #adj_matrix=adj_matrix[:500]
    return X, adj_matrix
    
def load_netsim(dataset_dir, dataset_file):
    # load the files
    np.random.seed(42)
    M=3
    data = np.load(os.path.join(dataset_dir, dataset_file + '.npz'))
    X = data['X']
    X = norm(X)
    #indices_to_remove=[0,5,10]
    #X=np.delete(X,indices_to_remove,axis=-1)
    adj_matrix = data['adj_matrix']
    D=adj_matrix.shape[1]
    fixed_remove = [0, 5, 10]
    remaining_indices = np.setdiff1d(np.arange(D), fixed_remove)
    random_remove = np.random.choice(remaining_indices, size=M, replace=False).tolist()
    indices_to_remove = random_remove
    X = np.delete(X, indices_to_remove, axis=-1)
    adj_matrix=np.delete(adj_matrix,indices_to_remove,axis=1)
    adj_matrix=np.delete(adj_matrix,indices_to_remove,axis=2)
    return X, adj_matrix
    
def load_lorenz(dataset_dir, dataset_file):
    # load the files
    data = np.load(os.path.join(dataset_dir, dataset_file + '.npz'))

    X = data['X']
    #X=X.transpose(0,2,1)
    #X = norm(X)
    adj_matrix = data['adj_matrix']
    #adj_matrxi=adj_matrix.transpose(0,2,1)
    print(adj_matrix.shape)
    return X, adj_matrix

def load_AQI(dataset_dir, dataset_file):
    # load the files
    X = np.load(os.path.join(dataset_dir, f'X_aqi_6month.npy'))
    #X = X[:, :, :20]
    #X=norm(X)
    #X = X - X.mean()
    D = X.shape[2]
    # we do not have the true adjacency matrix
    adj_matrix = np.zeros((X.shape[0], D, D))
    return X, adj_matrix


def load_traffic(dataset_dir, dataset_file):
    # load the files
    X = np.load(os.path.join(dataset_dir, f'traffic_gen_data.npy'))
    X = X[:, :, :20]
    X = X - X.mean()
    adj_matrix = np.load(os.path.join(dataset_dir, f'traffic_graph.npy'))
    adj_matrix = np.tile(adj_matrix, (X.shape[0], 1, 1))
    return X, adj_matrix


def load_medical(dataset_dir, dataset_file):
    # load the files
    X = np.load(os.path.join(dataset_dir, f'medical_gen_data.npy'))
    X = X[:, :, :20]
    X = X - X.mean()
    adj_matrix = np.load(os.path.join(dataset_dir, f'medical_graph.npy'))
    adj_matrix = np.tile(adj_matrix, (X.shape[0], 1, 1))
    return X, adj_matrix


def load_pm25(dataset_dir, dataset_file):
    # load the files
    X = np.load(os.path.join(dataset_dir, f'pm25_gen_data.npy'))
    X = X[:, :, :36]
    X = X - X.mean()
    adj_matrix = np.load(os.path.join(dataset_dir, f'pm25_graph.npy'))
    adj_matrix = np.tile(adj_matrix, (X.shape[0], 1, 1))
    return X, adj_matrix


def load_tm(dataset_dir, dataset_file):
    # load the files
    #X = np.load(os.path.join(dataset_dir, f'traffic_medical_data.npy'))
    data = np.load(os.path.join(dataset_dir, dataset_file + '.npz'))
    X=data["X"][:5]
    #X = X - X.mean()
    #X=norm(X)
    #adj_matrix = np.load(os.path.join(dataset_dir, f'traffic_medical_adj.npy'))
    adj_matrix=data["adj_matrix"][:5]
    return X, adj_matrix


def load_dream3_combined(dataset_dir, size):
    data = np.load(os.path.join(dataset_dir, f'combined_{size}.npz'))
    X = data['X']
    X = X - X.mean()
    adj_matrix = data['adj_matrix']
    return X, adj_matrix


def load_Macaque(dataset_dir, dataset_file):
    # load the files
    data = np.load(os.path.join(dataset_dir, dataset_file + '.npz'))
    X = data['data']
    adj_matrix = data['matrix']
    # adj_matrix = np.transpose(adj_matrix, (0, 2, 1))
    return X, adj_matrix


def load_yeast(dataset_dir, size, index):
    # load the files
    data = np.load(os.path.join(dataset_dir, f'yeast_{size}/yeast_{index}.npz'))
    X = data['X']
    X = X - X.mean()
    adj_matrix = data['adj_matrix']
    return X, adj_matrix


def load_ecoli(dataset_dir, size, index):
    # load the files
    data = np.load(os.path.join(dataset_dir, f'ecoli_{size}/ecoli_{index}.npz'))
    X = data['X']
    X = X - X.mean()
    adj_matrix = data['adj_matrix']
    return X, adj_matrix


def load_data(dataset, dataset_dir, config):
    if dataset in ['edges1', 'edges2', 'edges3', 'edges4']:
        X, adj_matrix = load_simdata(dataset_dir=dataset_dir, dataset_file=dataset)
        N = 10
        T = 100
        aggregated_graph = True
        # read lag from config file
        lag = int(config['lag'])
        data_dim = 1
        X = X.reshape(N, T, -1)
        X = np.expand_dims(X, axis=-1)
        adj_matrix = np.tile(adj_matrix, (X.shape[0], 1, 1))
        
    elif 'aqi' in dataset:
        X, adj_matrix = load_AQI(
            dataset_dir=dataset_dir, dataset_file=dataset)
        aggregated_graph = True
        # read lag from config file
        lag = int(config['lag'])
        data_dim = 1
        X = np.expand_dims(X, axis=-1)
        
    elif 'netsim' in dataset:
        X, adj_matrix = load_netsim(
            dataset_dir=dataset_dir, dataset_file=dataset)
        aggregated_graph = True
        # read lag from config file
        lag = int(config['lag'])
        data_dim = 1
        X = np.expand_dims(X, axis=-1)
    elif dataset == 'dream3_combined':
        dream3_size = int(config['dream3_size'])
        X, adj_matrix = load_dream3_combined(
            dataset_dir=dataset_dir, size=dream3_size)
        lag = int(config['lag'])
        data_dim = 1
        aggregated_graph = True
        X = np.expand_dims(X, axis=-1)
    elif dataset == 'yeast1':
        dream3_size = int(config['dream3_size'])
        X, adj_matrix = load_yeast(
            dataset_dir=dataset_dir, size=dream3_size, index=1)
        lag = int(config['lag'])
        data_dim = 1
        aggregated_graph = True
        X = np.expand_dims(X, axis=-1)
    elif dataset == 'yeast2':
        dream3_size = int(config['dream3_size'])
        X, adj_matrix = load_yeast(
            dataset_dir=dataset_dir, size=dream3_size, index=2)
        lag = int(config['lag'])
        data_dim = 1
        aggregated_graph = True
        X = np.expand_dims(X, axis=-1)
    elif dataset == 'yeast3':
        dream3_size = int(config['dream3_size'])
        X, adj_matrix = load_yeast(
            dataset_dir=dataset_dir, size=dream3_size, index=3)
        lag = int(config['lag'])
        data_dim = 1
        aggregated_graph = True
        X = np.expand_dims(X, axis=-1)
    elif dataset == 'ecoli1':
        dream3_size = int(config['dream3_size'])
        X, adj_matrix = load_ecoli(
            dataset_dir=dataset_dir, size=dream3_size, index=1)
        lag = int(config['lag'])
        data_dim = 1
        aggregated_graph = True
        X = np.expand_dims(X, axis=-1)
    elif dataset == 'ecoli2':
        dream3_size = int(config['dream3_size'])
        X, adj_matrix = load_ecoli(
            dataset_dir=dataset_dir, size=dream3_size, index=2)
        lag = int(config['lag'])
        data_dim = 1
        aggregated_graph = True
        X = np.expand_dims(X, axis=-1)
    elif dataset == 'Macaque':
        X, adj_matrix = load_Macaque(dataset_dir=dataset_dir, dataset_file=dataset)
        X = norm(X)
        aggregated_graph = True
        lag = int(config['lag'])
        data_dim = 1

    elif dataset in ["traffic_medical","causaltime_cluster_0","causaltime_cluster_1"]:
        X, adj_matrix = load_tm(dataset_dir=dataset_dir, dataset_file=dataset)
        aggregated_graph = True
        lag = int(config['lag'])
        data_dim = 1
        X = np.expand_dims(X, axis=-1)
    elif dataset == "traffic":
        X, adj_matrix = load_traffic(dataset_dir=dataset_dir, dataset_file=dataset)
        aggregated_graph = True
        lag = int(config['lag'])
        data_dim = 1
        X = np.expand_dims(X, axis=-1)
    elif dataset == "medical":
        X, adj_matrix = load_medical(dataset_dir=dataset_dir, dataset_file=dataset)
        aggregated_graph = True
        lag = int(config['lag'])
        data_dim = 1
        X = np.expand_dims(X, axis=-1)
    elif dataset == "pm25":
        X, adj_matrix = load_pm25(dataset_dir=dataset_dir, dataset_file=dataset)
        aggregated_graph = True
        lag = int(config['lag'])
        data_dim = 1
        X = np.expand_dims(X, axis=-1)
    elif dataset == 'MDD':
        X, adj_matrix = load_MDD(dataset_dir=dataset_dir, dataset_file=dataset)
        aggregated_graph = True
        lag = int(config['lag'])
        data_dim = 1
        X = np.expand_dims(X, axis=-1)
    elif dataset == "RestingLeft":
        X, adj_matrix = load_Resting(dataset_dir=dataset_dir, dataset_file=dataset)
        aggregated_graph = True
        lag = int(config['lag'])
        data_dim = 1
    elif dataset == "RestingRight":
        X, adj_matrix = load_Resting(dataset_dir=dataset_dir, dataset_file=dataset)
        aggregated_graph = True
        lag = int(config['lag'])
        data_dim = 1
    elif 'lorenz96' in dataset:
        X, adj_matrix = load_lorenz(dataset_dir=dataset_dir, dataset_file=dataset)
        aggregated_graph = True
        lag = int(config['lag'])
        data_dim = 1
        X = np.expand_dims(X, axis=-1)
    else:
        X, adj_matrix = load_synthetic_from_folder(
            dataset_dir=dataset_dir, dataset_file=dataset)
        #lag = adj_matrix.shape[1] - 1
        lag=int(config['lag'])
        data_dim = 1
        X = np.expand_dims(X, axis=-1)
        aggregated_graph = True
    print("Loaded data of shape:", X.shape)
    return X, adj_matrix, aggregated_graph, lag, data_dim
