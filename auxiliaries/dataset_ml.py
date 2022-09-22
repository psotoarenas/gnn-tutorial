# Standard python libraries.
import glob
import os
import shutil
import ssl
from six.moves import urllib
import errno
import zipfile
from collections import defaultdict
from math import inf

# General data science libraries.
import numpy as np

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

# set random seeds
SEED = 0
np.random.seed(SEED)

########################################################################################################################
# Dataset utilities.
########################################################################################################################


def parse_deployment(fn):
    # Parse the deployment identifier.
    deployment = int(fn.split("/")[-1].split("_")[-1].split(".")[0])
    return deployment


def parse_node_code(node_code):
    # Split the node code string. Example node codes: AP_A, STA_A1.
    parts = node_code.split("_")

    # Extract:
    #     - node type : access point (AP), or station (STA).
    #     - node id : unique node identifier within the deployment.
    node_type = parts[0]
    node_id = parts[1]

    # Extract the wlan identifier {A, B, .., L}. Note that the wlan
    # identifier determines the parent of the node in the graph.
    node_wlan = node_id[0]

    # Extract the wlan 'address', i.e., each station gets a unique
    # (integer) identifier in its wlan.
    node_wlan_addr = 0
    if node_type == "STA":
        node_wlan_addr = int(node_id[1:])

    return node_type, node_wlan, node_wlan_addr


def convert_airtime(airtime):
    airtime = np.array(airtime)
    air = np.zeros(8)  # max 8 channels are used
    air[:airtime.shape[0]] = airtime
    return np.mean(air)


def convert_interference(interference):
    interference = np.array(interference)
    interference = np.where(interference == inf, 0, interference)
    return np.mean(interference)

########################################################################################################################
# Node deployment input utilities.
########################################################################################################################


def read_nodes(fn):
    df = pd.read_csv(fn, sep=";")
    data = df.to_dict(orient="records")
    return data

########################################################################################################################
# edge deployment input utilities.
########################################################################################################################


def euclidean_distance(pos_a, pos_b):
    distance = np.linalg.norm(pos_a - pos_b)  # L2 norm
    return distance


########################################################################################################################
# Simulator output utilities.
########################################################################################################################


def read_list(fn):
    with open(fn, "r") as f:
        line = next(f)
        data = [float(x) for x in line.strip().split(",")]
    return data


def read_list_of_lists(fn):
    data = []
    with open(fn, "r") as f:
        line = next(f)
        tmp = line.strip().split(";")
        tmp = [x for x in tmp if x]
        for r in tmp:
            data.append([float(x) for x in r.split(',')])
    return data


def read_matrix(fn):
    data = []
    with open(fn, "r") as f:
        for line in f:
            line = line.strip().replace(';', '')
            row = [float(x) for x in line.split(',')]
            data.append(row)
    return data


########################################################################################################################
# Read the many many many custom files into a single data structure ...
########################################################################################################################


def read_dataset(path):
    # Dataset is organised per scenario.
    scenarios = {
        "train": [
            "sce1a",
            "sce1b",
            "sce1c",
            "sce2a",
            "sce2b",
            "sce2c"
        ],
        "test": [
            "test_sce1",
            "test_sce2",
            "test_sce3",
            "test_sce4",
        ]
    }

    # Dataset is stored per split in a dictionary, where the keys are (scenario, deployment) tuples.
    dataset = {
        'train': defaultdict(dict),
        'test': defaultdict(dict)
    }

    for split in ['train', 'test']:
        # Load input node files (deployments).
        nodes_path = os.path.join(path, split, 'input_node_files')
        for scenario in scenarios[split]:
            nodes_files = glob.glob(os.path.join(nodes_path, scenario, "*.csv"))
            for fn in sorted(nodes_files):
                deployment = parse_deployment(fn)
                data = read_nodes(fn)
                dataset[split][(scenario, deployment)]['devices'] = data
                dataset[split][(scenario, deployment)]['simulator'] = {}

        # Load simulator output files.
        simulations_path = os.path.join(path, split, 'output_simulator')
        for scenario in scenarios[split]:
            # Load airtime.
            for measurement in ['airtime']:
                measurement_files = glob.glob(
                    os.path.join(simulations_path, f"{scenario}_output", f"{measurement}_*.csv"))
                for fn in sorted(measurement_files):
                    deployment = parse_deployment(fn) - 1
                    data = read_list_of_lists(fn)
                    dataset[split][(scenario, deployment)]['simulator'][measurement] = data

            # Load RSSI, SINR, and throughput.
            for measurement in ['rssi', 'sinr', 'throughput']:
                measurement_files = glob.glob(
                    os.path.join(simulations_path, f"{scenario}_output", f"{measurement}_*.csv"))
                for fn in sorted(measurement_files):
                    deployment = parse_deployment(fn) - 1
                    data = read_list(fn)
                    if split == 'test' and measurement == 'throughput':
                        continue
                    dataset[split][(scenario, deployment)]['simulator'][measurement] = data

            # Load interference.
            for measurement in ['interference']:
                measurement_files = glob.glob(
                    os.path.join(simulations_path, f"{scenario}_output", f"{measurement}_*.csv"))
                for fn in sorted(measurement_files):
                    deployment = parse_deployment(fn) - 1
                    data = read_matrix(fn)
                    dataset[split][(scenario, deployment)]['simulator'][measurement] = data

    # Split the training dataset into a training and validation dataset. A fixed split has been created beforehand.
    train_split = pd.read_csv(os.path.join(path, 'train', 'train.csv'), header=None).values.tolist()
    valid_split = pd.read_csv(os.path.join(path, 'train', 'valid.csv'), header=None).values.tolist()

    train_split = set([tuple(row) for row in train_split])
    valid_split = set([tuple(row) for row in valid_split])

    train_dataset = {}
    valid_dataset = {}

    for (scenario, deployment), data in dataset['train'].items():
        if (scenario, deployment) in train_split:
            train_dataset[(scenario, deployment)] = data
        elif (scenario, deployment) in valid_split:
            valid_dataset[(scenario, deployment)] = data
        else:
            raise Exception(f'Scenario {scenario} and deployment {deployment} not found in splits.')

    dataset['train'] = train_dataset
    dataset['valid'] = valid_dataset

    return dataset

########################################################################################################################
# Put the data in a graph, without changing it ...
########################################################################################################################


def create_raw_graph(sample, scenario, deployment, n_id):
    # Mappings.
    wlan_to_node_id = {}

    # devices.
    devices = []

    # devices mask: indicated if a device is an AP (0) or a STA (1)
    device_mask = []

    # device features, targets, etc.
    device_features = []
    device_targets = []
    device_ap = []
    device_scenario = []
    device_deployment = []
    device_id = []

    # Station and access point features.
    sample_devices = sample['devices']
    sample_rssi = sample['simulator']['rssi']
    sample_sinr = sample['simulator']['sinr']
    sample_interference = sample['simulator']['interference']

    # Access point only features.
    sample_airtime = sample['simulator']['airtime']

    # Targets.
    if 'throughput' in sample['simulator']:
        # Targets (train)
        sample_throughput = sample['simulator']['throughput']
    else:
        # Dummy targets (test)
        sample_throughput = [-1 for _ in range(len(sample_devices))]

    k = 0
    for node_id, (node, rssi, sinr, throughput) in enumerate(
            zip(sample_devices, sample_rssi, sample_sinr, sample_throughput)):
        node_type, node_wlan, node_wlan_addr = parse_node_code(node["node_code"])

        # Nodes, features, and targets.
        devices.append(node_id)
        device_id.append(n_id)
        device_targets.append(throughput)
        # Complete list of features [node_type, x(m), y(m), primary_channel, min_channel_allowed, max_channel_allowed,
        # distance(AP-STA only), SINR, airtime, interference, RSSI]
        features = [
            node['node_type'],
            node['x(m)'],
            node['y(m)'],
            node['primary_channel'],
            node['min_channel_allowed'],
            node['max_channel_allowed']
        ]

        # Links between stations and access points.

        if node_type == "AP":
            # Register access point.
            ap_id = n_id
            wlan_to_node_id[node_wlan] = node_id
            device_mask.append(0)
            # sinr=0 as feature for APs
            features.append(0)
            # airtime (mean) as feature for APs and STAs
            airtime = sample_airtime[k]
            airtime = convert_airtime(airtime)
            features.append(airtime)
            # interference (mean) as feature for APs and STAs
            interference = sample_interference[k]
            interference = convert_interference(interference)
            features.append(interference)
            k += 1
            # rssi=0 as feature for APs
            features.append(0)
            # distance=0 as feature for APs
            features.append(0)
            # Bandwidth = 20 MHz per used channel
            bw = (node['max_channel_allowed'] - node['min_channel_allowed'] +1) * 20
            features.append(bw)

        if node_type == "STA":
            # Create an edge between the AP and STA.
            ap_node_id = wlan_to_node_id[node_wlan]
            device_mask.append(1)
            # sanity check
            if np.isnan(sinr):
                sinr = 0
            # sinr as node feature for STAs
            features.append(sinr)
            # airtime (mean) as feature for APs and STAs
            features.append(airtime)
            # interference (mean) as feature for APs and STAs
            features.append(interference)
            # rssi as node feature for STAs
            features.append(rssi)
            # Distance between STA - AP
            pos_ap = np.asarray([sample_devices[ap_node_id]['x(m)'], sample_devices[ap_node_id]['y(m)']])
            pos_sta = np.asarray([node['x(m)'], node['y(m)']])
            distance = euclidean_distance(pos_ap, pos_sta)
            features.append(distance)
            # Bandwidth = 20 MHz per used channel, same as AP
            features.append(bw)

        # Store the node id of the AP associated with the STA.
        # Note: APs are associated with themselves.
        device_ap.append(ap_id)

        # Store node features
        device_features.append(features)
        device_scenario.append(scenario)
        device_deployment.append(deployment)

        # Next node_id
        n_id += 1

    # Merge all info.
    graph = {
        # Devices.
        "devices": devices,
        # Features.
        "device_features": device_features,
        # Device targets.
        "device_targets": device_targets,
        # Utilities: masks and associations.
        "device_id": device_id,
        "device_mask": device_mask,
        "device_ap": device_ap,
        "device_scenario": device_scenario,
        "device_deployment": device_deployment,
    }

    return graph, n_id

########################################################################################################################
# Pre-process Data
########################################################################################################################


def create_preprocessors(graphs):
    # Extract coordinates.
    x_coord, y_coord = [], []
    for g in graphs:
        x, y = zip(*[(d[1], d[2]) for d in g['device_features']])
        x_coord.extend(x)
        y_coord.extend(y)

    x_coord = np.array(x_coord).reshape(-1, 1)
    y_coord = np.array(y_coord).reshape(-1, 1)

    # Fit standard scalers.
    standard_scaler_x = StandardScaler()
    standard_scaler_x.fit(x_coord)

    standard_scaler_y = StandardScaler()
    standard_scaler_y.fit(y_coord)

    # Fit range scalers.
    range_scaler_x = MinMaxScaler()
    range_scaler_x.fit(x_coord)

    range_scaler_y = MinMaxScaler()
    range_scaler_y.fit(y_coord)

    # Extract channel configuration info.
    channel_info = []
    for g in graphs:
        c = [[d[3], d[4], d[5]] for d in g['device_features']]
        channel_info.extend(c)
    channel_info = np.array(channel_info)
    channel_info = pd.DataFrame(channel_info, columns=['primary_channel', 'min_channel_allowed', 'max_channel_allowed'])

    # Create a mapping from (primary, min, max) tuples to an integer id.
    channel_configs = {}
    for i, channel_config in enumerate(channel_info.drop_duplicates().values):
        channel_configs[tuple(channel_config)] = i

    # Fit a channel config (one-hot) encoder.
    channel_config_ids = np.array(list(channel_configs.values())).reshape(-1, 1)
    channel_config_encoder = OneHotEncoder(sparse=False)
    channel_config_encoder.fit(channel_config_ids)

    # Transform channel config ids.
    channel_config_encoder.transform(channel_config_ids)

    # Extract everything else
    sinr, airtime, interference, rssi, distance, bw = [], [], [], [], [], []
    for g in graphs:
        s, a, i, r, ds, b = zip(*[(d[6], d[7], d[8], d[9], d[10], d[11]) for d in g['device_features']])
        sinr.extend(s)
        airtime.extend(a)
        interference.extend(i)
        rssi.extend(r)
        distance.extend(ds)
        bw.extend(b)

    sinr = np.array(sinr).reshape(-1, 1)
    airtime = np.array(airtime).reshape(-1, 1)
    interference = np.array(interference).reshape(-1, 1)
    rssi = np.array(rssi).reshape(-1, 1)
    distance = np.array(distance).reshape(-1, 1)
    bw = np.array(bw).reshape(-1, 1)

    # Fit standard scalers.
    standard_scaler_sinr = StandardScaler()
    standard_scaler_sinr.fit(sinr)

    standard_scaler_airtime = StandardScaler()
    standard_scaler_airtime.fit(airtime)

    standard_scaler_interference = StandardScaler()
    standard_scaler_interference.fit(interference)

    standard_scaler_rssi = StandardScaler()
    standard_scaler_rssi.fit(rssi)

    standard_scaler_distance = StandardScaler()
    standard_scaler_distance.fit(distance)

    standard_scaler_bw = StandardScaler()
    standard_scaler_bw.fit(bw)

    # Fit range scalers.
    range_scaler_sinr = MinMaxScaler()
    range_scaler_sinr.fit(sinr)

    range_scaler_airtime = MinMaxScaler()
    range_scaler_airtime.fit(airtime)

    range_scaler_interference = MinMaxScaler()
    range_scaler_interference.fit(interference)

    range_scaler_rssi = MinMaxScaler()
    range_scaler_rssi.fit(rssi)

    range_scaler_distance = MinMaxScaler()
    range_scaler_distance.fit(distance)

    range_scaler_bw = MinMaxScaler()
    range_scaler_bw.fit(bw)

    preprocessors = {
        'x': {
            'standard': standard_scaler_x,
            'range': range_scaler_x,
        },
        'y': {
            'standard': standard_scaler_y,
            'range': range_scaler_y,
        },
        'channel_info': {
            'categorical': channel_configs,
            'one_hot': channel_config_encoder,
        },
        'sinr': {
            'standard': standard_scaler_sinr,
            'range': range_scaler_sinr,
        },
        'airtime': {
            'standard': standard_scaler_airtime,
            'range': range_scaler_airtime,
        },
        'interference': {
            'standard': standard_scaler_interference,
            'range': range_scaler_interference,
        },
        'rssi': {
            'standard': standard_scaler_rssi,
            'range': range_scaler_rssi,
        },
        'distance': {
            'standard': standard_scaler_distance,
            'range': range_scaler_distance,
        },
        'bw': {
            'standard': standard_scaler_bw,
            'range': range_scaler_bw,
        }
    }

    return preprocessors


def preprocess_graph(graph, preprocessors):
    # Pre-process device features
    device_features = graph['device_features']

    # Pre-process node type.
    node_type = np.array([d[0] for d in device_features]).reshape(-1, 1)

    # Pre-process coordinates.
    x_coord, y_coord = zip(*[(d[1], d[2]) for d in device_features])
    x_coord = np.array(x_coord).reshape(-1, 1)
    y_coord = np.array(y_coord).reshape(-1, 1)

    x_coord = preprocessors['x']['range'].transform(x_coord)
    y_coord = preprocessors['y']['range'].transform(y_coord)

    # Pre-process channel configuration: categorical encoding (step 1).
    channel_configs = [(d[3], d[4], d[5]) for d in graph['device_features']]
    channel_configs = [preprocessors['channel_info']['categorical'][c] for c in channel_configs]

    # Pre-process channel configuration: one-hot encoding (step 2).
    channel_configs = np.array(channel_configs).reshape(-1, 1)
    channel_configs = preprocessors['channel_info']['one_hot'].transform(channel_configs)

    # Pre-process everything else
    sinr, airtime, interference, rssi, distance, bw = zip(*[(d[6], d[7], d[8], d[9], d[10], d[11]) for d in device_features])

    # Pre-process sinr
    sinr = np.array(sinr).reshape(-1, 1)
    sinr = preprocessors['sinr']['range'].transform(sinr)

    # Pre-process airtime
    airtime = np.array(airtime).reshape(-1, 1)
    airtime = preprocessors['airtime']['range'].transform(airtime)

    # Pre-process interference
    interference = np.array(interference).reshape(-1, 1)
    interference = preprocessors['interference']['range'].transform(interference)

    # Pre-process rssi
    rssi = np.array(rssi).reshape(-1, 1)
    rssi = preprocessors['rssi']['range'].transform(rssi)

    # Pre-process distance
    distance = np.array(distance).reshape(-1, 1)
    distance = preprocessors['distance']['range'].transform(distance)

    # Pre-process bandwidth
    bw = np.array(bw).reshape(-1, 1)
    bw = preprocessors['bw']['range'].transform(bw)

    # Create preprocessed feature vector.
    device_features_processed = [
        node_type,
        x_coord,
        y_coord,
        channel_configs,
        sinr,
        airtime,
        interference,
        rssi,
        distance,
        bw
    ]
    device_features_processed = np.concatenate(device_features_processed, axis=1)

    # Update graph.
    graph['device_features'] = device_features_processed

    return graph

########################################################################################################################
# Final Data
########################################################################################################################


def split_save_dataset(split, data_list, label_list, scenario_list, deployment_list, id_list, ap_list, path):
    datasets = {}
    labels = {}
    scenarios = {}
    deployments = {}
    ids = {}
    aps = {}

    # convert lists to a numpy array
    datasets['all'] = np.vstack(data_list)
    labels['all'] = np.hstack(label_list)
    scenarios['all'] = np.hstack(scenario_list)
    deployments['all'] = np.hstack(deployment_list)
    ids['all'] = np.hstack(id_list)
    aps['all'] = np.hstack(ap_list)

    # divide the entire dataset in APs
    rows = np.where(datasets['all'][:, 0] == 0)[0]  # finds the indexes of the APs
    datasets['ap'] = datasets['all'][rows]
    labels['ap'] = labels['all'][rows]
    scenarios['ap'] = scenarios['all'][rows]
    deployments['ap'] = deployments['all'][rows]
    ids['ap'] = ids['all'][rows]
    aps['ap'] = aps['all'][rows]

    # divide the entire dataset in STAs
    rows = np.where(datasets['all'][:, 0] == 1)[0]  # finds the indexes of the STAs
    datasets['sta'] = datasets['all'][rows]
    labels['sta'] = labels['all'][rows]
    scenarios['sta'] = scenarios['all'][rows]
    deployments['sta'] = deployments['all'][rows]
    ids['sta'] = ids['all'][rows]
    aps['sta'] = aps['all'][rows]

    # # extra processing to take features except node_type
    # datasets['all'] = datasets['all'][:, 1:]
    # datasets['ap'] = datasets['ap'][:, 1:]
    # datasets['sta'] = datasets['sta'][:, 1:]

    # save datasets
    path = os.path.join(path, split)
    os.makedirs(path, exist_ok=True)
    output_files = []
    for key, data in datasets.items():
        outfile = os.path.join(path, 'data_%s.npz' % key)
        if key == 'ap':
            np.savez(
                outfile,
                x_ap=data,
                y_ap=labels[key],
                sce_ap=scenarios[key],
                dpl_ap=deployments[key],
                ids_ap=ids[key],
                aps_ap=aps[key]
            )
        elif key == 'sta':
            np.savez(
                outfile,
                x_sta=data,
                y_sta=labels[key],
                sce_sta=scenarios[key],
                dpl_sta=deployments[key],
                ids_sta=ids[key],
                aps_sta=aps[key]
            )
        else:
            np.savez(
                outfile,
                x_all=data,
                y_all=labels[key],
                sce_all=scenarios[key],
                dpl_all=deployments[key],
                ids_all=ids[key],
                aps_all=aps[key]
            )
        output_files.append(outfile)
    return output_files


########################################################################################################################
# Keras dataset implementation.
########################################################################################################################


class NDTDataset:
    # get a direct link to download a file from Google Drive by replacing the shareable url with
    # https://drive.google.com/uc?export=download&id=DRIVE_FILE_ID and the corresponding file id.
    dataset_url = "https://drive.google.com/uc?export=download&id=14rD7TjcSLw6Qxouk2rdgW72Op-byfuU_"

    def __init__(self, root):
        self.raw_dir = os.path.join(root, 'raw')
        self.processed_dir = os.path.join(root, 'processed')
        if not os.path.exists(self.raw_dir):
            self.makedirs(self.raw_dir)
        if not os.path.exists(self.processed_dir):
            self.makedirs(self.processed_dir)
        self.download()
        self.process()

    def download_url(self, url, folder, log=True):
        # Taken from
        # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/download.html#download_url

        filename = url.rpartition('/')[2]
        path = os.path.join(folder, filename)

        if os.path.exists(path):  # pragma: no cover
            if log:
                print('Using exist file', filename)
            return path

        if log:
            print('Downloading', url)

        self.makedirs(folder)

        context = ssl._create_unverified_context()
        data = urllib.request.urlopen(url, context=context)

        with open(path, 'wb') as f:
            f.write(data.read())

        return path

    def makedirs(self, path):
        try:
            os.makedirs(os.path.expanduser(os.path.normpath(path)))
        except OSError as e:
            if e.errno != errno.EEXIST and os.path.isdir(path):
                raise e

    def maybe_log(self, path, log=True):
        # From https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/data/extract.py
        if log:
            print('Extracting', path)

    def extract_zip(self, path, folder, log=True):
        # From https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/data/extract.py
        self.maybe_log(path, log)
        with zipfile.ZipFile(path, 'r') as f:
            f.extractall(folder)

    def download(self):
        # Prepare raw data directory.
        shutil.rmtree(self.raw_dir)

        # Download and extract the dataset.
        path = self.download_url(self.dataset_url, self.raw_dir)
        self.extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        node_id = 0
        datasets = read_dataset(self.raw_dir)

        preprocessors = None
        output_files = {}
        for split in ['train', 'valid', 'test']:
            print(f"Processing {split} split.")
            # Read data for each split into a huge `Data` list.
            graphs = []

            for (scenario, deployment), sample in datasets[split].items():
                graph, node_id = create_raw_graph(sample, scenario, deployment, node_id)
                graphs.append(graph)

            if split == 'train':
                # Analyse data and fit preprocessors (e.g., scalers, encoders).
                preprocessors = create_preprocessors(graphs)

            # Pre-process graph (feature scaling and encoding).
            data_list = []
            label_list = []
            scenario_list = []
            deployment_list = []
            device_id_list = []
            device_ap_list = []
            for graph in graphs:
                graph = preprocess_graph(graph, preprocessors)
                data_list.append(graph['device_features'].tolist())
                label_list.append(graph['device_targets'])
                scenario_list.append(graph['device_scenario'])
                deployment_list.append(graph['device_deployment'])
                device_id_list.append(graph['device_id'])
                device_ap_list.append(graph['device_ap'])

            # Split entire dataset in dataset_sta and dataset_ap and save
            output_files[split] = split_save_dataset(split,data_list, label_list, scenario_list, deployment_list, device_id_list, device_ap_list, self.processed_dir)
        return output_files