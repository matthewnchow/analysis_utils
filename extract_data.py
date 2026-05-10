import numpy as np 
import matplotlib.pyplot as plt 
import os
import re
from collections import Counter


# Dictionary for converting names of evaluations 
RAW_EVALS = {'check_1':102, 'check_2':202, 'check_3':46, # Check results
            'det_1':101, 'det_2':201, 'det_3':45, 'det_13':13, # Det results? TODO: check that these numbers are right. 
            'load_1':1, 'load_2':2, 'load_3':3, # Load counters
            'check_1_count':513, 'check_2_count':514, 'check_3_count':515, # Check counters
            'bkg_1':769, 'bkg_2':770, 'bkg_3':771, # Background counters
            'det_1_count':1056, 'det_2_count':1057, 'det_3_count':1062, # Detection counters? TODO: check these are right. BL changed last one from 1058 to 1062
            'reps': 1 # Number of repetitions result
            }
ENCODING = 'iso-8859-1' # Note, not usually needed to specify, but for some reason auto detection not working well on Matt's linux machine.

def parse_header(file_path):
    """ Grabs the header of an ioncontrol data file and returns it as a dictionary.
    Inputs: file_path: string  (path to file)
     :returns  header : dictionary : keys are strings for ppp parameters and things like ColumnSpec)"""
    header = {}
    with open(file_path, encoding=ENCODING) as f:
        line = f.readline()
        while line[0] == "#":
            if "Element name" in line:
                name = re.findall("\"(.*?)\"", line)[0]
                val = re.findall(">(.*?)<", line)
                if len(val)>0:
                    val = val[0]
                    header[name] = val
            elif "ColumnSpec" in line:
                col_spec = re.findall(">(.*?)<", line)
                col_spec = col_spec[0].split(", ")
                header["ColumnSpec"] = col_spec
            line = f.readline()
    return header
    

def parse_data(file_path, columns=None, rows=[], get_header=False, raw_file=False, nan_to_zero=True):
    """Get data from an ioncontrol data file.
    get_header parses through comments at beginning and returns dictionary of parameters
    data_slice returns only the parts we care about

    Inputs:
        1.) file_path : string  : path to the data file
        2a.) columns : array-like of ints : Column indices of the data file to be returned. Uses 0:end by default
            2b) Other option for columns is list-like of strings with the names of the desired columns
                'x' not necessary to include because it is always included
        3.) rows : array-like of ints: Row indices of the data file to be returned. Uses 0:end by default
        4.) get_header : Boolean : determines if we also get the header of the file
        5.) raw_file : Used for "raw" data files out of ioncontrol.

    Outputs:
        data : np array : Data from file
        header (optional) : dictionary : Header of the same file

    Note: Still inefficiently reading entire data file, instead of just getting rows and cols
    """
    try: 
        if raw_file:
            pass
        else:
            # print(columns)
            if columns is not None and len(columns) > 0:
                if isinstance(columns[0], str):
                    col_spec = parse_header(file_path)['ColumnSpec']
                    temp = [col_spec.index('x')]
                    for col in columns:
                        if col != 'x':
                            if col in col_spec:
                                temp.append(col_spec.index(col))
                            else:
                                print("Could not find column : %s" % col)
                    columns = temp
            else: # just get the x data
                col_spec = parse_header(file_path)['ColumnSpec']
                columns = [col_spec.index('x')]
            data = np.loadtxt(file_path, usecols=columns, encoding=ENCODING)  # grabs relevant columns only, no headers
            if nan_to_zero:
                data[np.isnan(data)] = 0
            if len(rows) > 0:
                data = data[rows, :]
            if get_header:
                header = parse_header(file_path)
                return data, header
            return data

    except OSError as e:
        print(e)
        print("Check data file path")
        return None


def process_data(data_array, data_type=None, shots=50, sorted=True, z=1.96):
    """Combines all like x values and returns mean and error of y values associated with that x value.
    Inputs:
        1.) data_array : 2D np array
            Requires x array to be the first column
            Y values are the subsequent columns
        2.) data_type : string : used to decide what processing to do on the data
                Options:
                    a.) no strings match : regular old mean and std
                    b.) state : assumes binomial rate data such as threshold data
                                (ioncontrol puts #success/total in state column)
                                requires shots
                    c.) state_raw : same as above, but taking number of events instead of rate.
                                (ioncontrol puts #success in raw column)
                                requires shots
                    d.) counts : assumes Poisson data
        3.) shots : int : Used to determine number of trials for some types of errors
        4.) sorted : Boolean : Sort the array based on the x values before returning it
        5.) z : number : (from wiki) "The 1-0.5alpha quantile of a standard normal distribution
            corresponding to the target error rate alpha"
            z=1.96 for 95% confidence interval
    :returns : 2D np array : processed data
        Columns:  (Ny = number of y columns)
            0) X values
            1:Ny+1) Avg Y values
            Ny+1:2Ny+1) Error for each Y

    Doesn't work yet with  y values of different data_types"""
    unique = Counter(data_array[:, 0])
    xs = list(unique.keys())
    for x in xs:
        unique[x] = []  # Replace counter with array of empty lists
    for i in range(len(data_array)):
        # Loop through data array and add all data corresponding to a certain x to dictionary entry
        unique[data_array[i, 0]].append(data_array[i, 1:])
    processed = np.zeros((len(unique), 1 + (len(data_array[0])-1) * 2))  # X, Y1_mean, Y2_mean,... Y1_std, Y2_std, ... 
    for i in range(len(xs)):
            yi = np.array(unique[xs[i]])

            if data_type == "state_raw":
                # Calculates errors with the Wald method for Binomial Distributions.
                # Note that z=1.96 is used for confidence interval of 95%
                means = np.mean(yi, 0)/shots
                errs = z*np.sqrt(means * (1-means)/(shots * len(yi)))     
            elif data_type == "state":
                means = np.mean(yi, 0)
                errs = z*np.sqrt(means * (1-means)/(shots * len(yi)))   # Wald method again.
            elif data_type == "counts":  # Should be used with Poisson data, such as photon counts
                means = np.mean(yi, 0)
                # errs = np.sqrt(np.sum(yi, 0))/np.sum(yi, 0)
                errs = np.sqrt(means/(shots*len(yi)))  # sqrt(lambda/N) for a Poisson process.
            else: 
                means = np.mean(yi, 0)
                errs = np.std(yi, 0)

            processed[i] = np.concatenate(([xs[i]], means, errs))
    if sorted:
        return processed[np.argsort(processed[:,0])]
    return processed
    
    
def parse_and_process(file_path, data_type='state', columns=['state_1', 'state_2'], rows=[], experiments=None, z=1.96):
    """ Wraps together parse_data and process_data. See those functions for IO information. Returns processed means and errors for each column. """
    if experiments is None:
        header = parse_header(file_path)
        experiments = int(header['experiments'])
    data_raw = parse_data(file_path, columns=columns, rows=rows, get_header=False)
    return process_data(data_raw, data_type=data_type, shots=experiments, z=z)


def parse_raw(file_path, skip_lines=None, end_line=None, get_counters=['check_3'], fast_parse=True):
    """ Parses ioncontrol 'raw' format datafiles. 
    Each line in the raw file is a single scanpoint. I think they're just recorded in the order they come in - MC.
    Returns: data : list of lists with format   
        0: list of scan variable values
        1: list of arrays (one for each scan point) containing counters[0] values
        ...
        len(counters): list of arrays (one for each scan point) containing counters[len(counters)-1] values
    """
    data = [[] for i in range(len(get_counters)+1)]
    scan_var = []
    with open(file_path, 'r', encoding=ENCODING) as f:
        i = 0
        for line in f:
            i += 1
            if skip_lines is not None and i <= skip_lines:
                continue
            elif end_line is not None and i > end_line:
                break
            if fast_parse:
                """ Manually parse through each line, looking for particular start flags 
                that are associated with each counter and the scan variable. This allows us to 
                parse only necessary parts of the line. MC 2022-03-17"""
                scan_var_flag = "null, null," # String that precedes the scan variable entry.
                scan_var_start_idx = line.find(scan_var_flag)+len(scan_var_flag)
                scan_var_end_idx = line.find(",", scan_var_start_idx)
                scan_var.append(float(line[scan_var_start_idx:scan_var_end_idx]))
                for j in range(len(get_counters)):
                    flag = r'"%d": ' % RAW_EVALS[get_counters[j]]
                    start_idx = line.find(flag)+len(flag)
                    end_idx = line.find("]", start_idx)+1
                    counts = eval(line[start_idx:end_idx])
                    data[j+1].append(np.array(counts))
            else:
                """ Old way of doing parsing that evaluates the entire line and then selects
                relevant portions to return. Much slower, but tested extensively. """
                line = line.replace("true", "True")
                line = line.replace("false", "False")
                line = line.replace("null", "None")
                line = eval(line)
                scan_var.append(line[3])
                counters = line[0]  # dictionary for all counters 
                results = line[9] # dictionary for all the results
                all_counters = {**counters, **results}
                if True: # TODO: add some filter conditions here.  
                    for j in range(len(get_counters)):
                        if get_counters[j] in RAW_EVALS.keys():
                            data[j+1].append(np.array(all_counters[str(RAW_EVALS[get_counters[j]])]))
                        else:
                            data[j+1].append(np.array(all_counters[str(get_counters[j])]))
    data[0] = scan_var
    return data
    
    
def collect_same_x(x_lst, y_lst):
    xs = list(Counter(x_lst).keys())    # Get unique x values
    result = {x:[] for x in xs}         # Make dict of empty lists, with x value as the key
    for i in range(len(x_lst)):
        # Loop through data array and add all data corresponding to a certain x to dictionary entry
        result[x_lst[i]] += list(y_lst[i])
    return result
    

def shots_per_point(datafile, xs=None, experiments=None, experiments_key='experiments', sorted=True):
    """ Function to return the total number of shots per point for each x value.
    Can pass an array of x values (xs) and the number of reps per x value (experiments) to 
    avoid calling parse_header and parse_data more than once. """
    if experiments is None: experiments = int(parse_header(datafile)[experiments_key])
    if xs is None: xs = parse_data(datafile)
    x_count = Counter(xs)
    keys = list(x_count.keys())
    shots = np.array([x_count[k]*experiments for k in keys])
    if sorted: shots = shots[np.argsort(keys)]
    return shots


if __name__ == "__main__":
    # # root = os.getcwd()
    # root = "\\\\snl\\Mesa\\Projects\\sigma-gc\\7-AdvancedSensingEntanglement\\Experiment\\important_data"
    # # print(root + "\\" + "test_data")

    # Example of how to use the functions in this file
    root = "C:\\Users\\Public\\Documents\\experiments\\AI\\2020\\2020_08\\2020_08_26\\"
    testfile = root + "load_fd1_021"
    
    # You can use these lines to go through each step of parsing and processing
    d1_header = parse_header(testfile)
    experiments = int(d1_header['experiments'])
    spec = d1_header['ColumnSpec']
    cols = [spec.index('x'), spec.index('thresh_1'), spec.index('thresh_2')]
    d1_data_raw = parse_data(testfile, columns=cols, get_header=False)
    d1_data = process_data(d1_data_raw, data_type="state", shots=experiments)
    
    # Or you can use this line, which is equivalent to the above 6 lines
    d1_data = parse_and_process(testfile, columns=["thresh_1", "thresh_2"])

    plt.errorbar(d1_data[:, 0], d1_data[:, 1], d1_data[:, 3], marker='.', linestyle='', color='r', label="Atom 1")
    plt.errorbar(d1_data[:, 0], d1_data[:, 2], d1_data[:, 4], marker='.', linestyle='', color='b', label="Atom 2")
    plt.legend()
    plt.show()
