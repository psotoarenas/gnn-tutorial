import zipfile
import argparse
import os
import shutil

########################################################################################################################
# Command line arguments.
########################################################################################################################

parser = argparse.ArgumentParser(description='Pre-process the original dataset.')
parser.add_argument('--dir', default='./', help='full path where the data from zenodo is.')
args = parser.parse_args()
data_dir = args.dir

########################################################################################################################
# Helpers.
########################################################################################################################


def extract_original_files(files_dir):
    # filter only .zip files
    files = [f for f in os.listdir(files_dir) if f.endswith(".zip")]

    # clean previous executions
    for mode in ["train", "test"]:
        path = os.path.join(files_dir, mode)
        # remove dir if exists
        if os.path.exists(path):
            shutil.rmtree(path)

    # extract .zip files
    for f in files:
        path_to_file = os.path.join(files_dir, f)
        print(f"Processing file: {path_to_file}")
        with zipfile.ZipFile(path_to_file, 'r') as zip_ref:
            # identify if file is from the train or the test split
            if f.endswith("_test.zip"):
                mode = "test"
                test_path = os.path.join(files_dir, mode)
            else:
                mode = "train"
                train_path = os.path.join(files_dir, mode)

            # identify if file is from the input or the output
            if f.startswith("input"):
                folder_name = f.split(".")[0]
                folder_name = folder_name.split("_")[:3]
            else:
                folder_name = f.split(".")[0]
                folder_name = folder_name.split("_")[:2]
            folder_name = "_".join(folder_name)

            extract_dir = os.path.join(files_dir, mode, folder_name)
            print(f"Extracting file to: {extract_dir}")
            zip_ref.extractall(extract_dir)
    return train_path, test_path


def parse_output_simulator(out_dir, mode):
    out_dir = os.path.join(out_dir, "output_simulator")
    # filter only .txt files
    files = [f for f in os.listdir(out_dir) if f.endswith(".txt")]

    for f in files:
        # get the scenario
        scenario = f.split("_")[-1]
        scenario = scenario.split(".")[0]
        if mode == "train":
            scenario = scenario + "_output"
        else:
            scenario = "test_" + scenario + "_output"

        # clean previous data and create folder
        scenario_path = os.path.join(out_dir, scenario)
        if os.path.exists(scenario_path):
            shutil.rmtree(scenario_path)
        os.makedirs(scenario_path)
        path_to_file = os.path.join(out_dir, f)
        print(f"Parsing file: {path_to_file}")

        with open(path_to_file, "r") as results:
            count = 0
            for line in results:
                if line.startswith(' KOMONDOR SIMULATION'):
                    # the respective deployment file is taken from the KOMONDOR SIMULATION line
                    input_file = line.split("\'")[1]
                    input_file = input_file.split('_')
                    input_file = '_'.join(input_file[1:])
                    deployment = input_file.split('.')[0]
                    deployment = int(deployment.split("_")[-1]) + 1
                if line.startswith('{'):
                    # the output of the simulation is reported between brackets
                    count += 1
                    if count == 1:
                        # the first line of the output is the aggregated throughput per AP and per STA connected to the AP - fixed size [#APs + #STAs] - list
                        throughput = line.rstrip().rstrip('}').lstrip('{')
                        # save the file
                        fn = 'throughput_' + str(deployment) + '.csv'
                        variable_path = os.path.join(scenario_path, fn)
                        with open(variable_path, "w+") as csv_file:
                            csv_file.write(throughput)
                            csv_file.write('\n')
                    if count == 2:
                        # the second line of the output is the airtime per set of selected channels - variable size
                        airtime = line.rstrip().rstrip('}').lstrip('{')
                        # save the file
                        fn = 'airtime_' + str(deployment) + '.csv'
                        variable_path = os.path.join(scenario_path, fn)
                        with open(variable_path, "w+") as csv_file:
                            csv_file.write(airtime)
                            csv_file.write('\n')
                    if count == 3:
                        # the third line of the output is the RSSI values from AP to STA - fixed size [#APs + #STAs] - list
                        rssi = line.rstrip().rstrip('}').lstrip('{')
                        # save the file
                        fn = 'rssi_' + str(deployment) + '.csv'
                        variable_path = os.path.join(scenario_path, fn)
                        with open(variable_path, "w+") as csv_file:
                            csv_file.write(rssi)
                            csv_file.write('\n')
                    if count == 4:
                        # the fourth line of the output is the interference map - fixed size [#APs, #APs] - matrix
                        interference = line.rstrip().lstrip('{')
                        # save the file
                        fn = 'interference_' + str(deployment) + '.csv'
                        variable_path = os.path.join(scenario_path, fn)
                        with open(variable_path, "w+") as csv_file:
                            csv_file.write(interference)
                            csv_file.write('\n')
                            processing = True
                            while processing:
                                next_line = next(results)
                                next_line = next_line.rstrip()
                                if next_line.endswith("}"):
                                    processing = False
                                    next_line = next_line.rstrip('}')
                                csv_file.write(next_line)
                                csv_file.write('\n')
                    if count == 5:
                        # the fifth line of the output is the SINR values from AP to STA - fixed size [#APs + #STAs] - list
                        sinr = line.rstrip().rstrip('}').lstrip('{')
                        # save the file
                        fn = 'sinr_' + str(deployment) + '.csv'
                        variable_path = os.path.join(scenario_path, fn)
                        with open(variable_path, "w+") as csv_file:
                            csv_file.write(sinr)
                            csv_file.write('\n')
                        count = 0
    return


def change_dir_name(path):
    test_path = os.path.join(path, "input_node_files")
    for directory in os.listdir(test_path):
        final_name = "test_sce" + directory.split("_")[1]
        os.rename(os.path.join(test_path, directory), os.path.join(test_path, final_name))
    return


def remove_script_files(path):
    output_path = os.path.join(path, "output_simulator")
    files = [f for f in os.listdir(output_path) if f.endswith(".txt")]
    for fle in files:
        os.remove(os.path.join(output_path, fle))
    return


########################################################################################################################
# Main.
########################################################################################################################

train_path, test_path = extract_original_files(files_dir=data_dir)
# parse train output
parse_output_simulator(train_path, "train")
# parse test output
parse_output_simulator(test_path, "test")
# change input_node_files from test set more accordingly.
change_dir_name(test_path)
# remove unparsed files
remove_script_files(test_path)
remove_script_files(train_path)
# zip parsed train and test splits: run in the terminal zip -r dataset.zip test train


