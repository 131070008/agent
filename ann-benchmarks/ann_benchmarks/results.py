import json
import os
import re
import traceback
from typing import Any, Optional, Set, Tuple, Iterator
import h5py

from ann_benchmarks.definitions import Definition


def get_thread_suffix() -> str:
    threads = os.environ.get("OMP_NUM_THREADS", "").strip()
    if not threads:
        return ""
    match = re.match(r"(\d+)", threads)
    return f"_{match.group(1)}t" if match else ""


def build_result_filepath(dataset_name: Optional[str] = None,
                          count: Optional[int] = None,
                          definition: Optional[Definition] = None,
                          query_arguments: Optional[Any] = None,
                          batch_mode: bool = False,
                          output_dir: Optional[str] = None) -> str:
    """
    Constructs the filepath for storing the results.

    Args:
        dataset_name (str, optional): The name of the dataset.
        count (int, optional): The count of records.
        definition (Definition, optional): The definition of the algorithm.
        query_arguments (Any, optional): Additional arguments for the query.
        batch_mode (bool, optional): If True, the batch mode is activated.

    Returns:
        str: The constructed filepath.
    """
    if output_dir:
        d = [output_dir]
    else:
        d = ["results"]
        if dataset_name:
            d.append(dataset_name)
        if count:
            d.append(str(count))
        if definition:
            d.append(definition.algorithm + ("-batch" if batch_mode else ""))
    if definition:
        data = definition.arguments + query_arguments
        basename = re.sub(r"\W+", "_", json.dumps(data, sort_keys=True)).strip("_")
        if output_dir:
            basename += get_thread_suffix()
        d.append(basename + ".hdf5")
    return os.path.join(*d)


def store_results(dataset_name: str, count: int, definition: Definition, query_arguments:Any, attrs, results, batch, output_dir: Optional[str] = None):
    """
    Stores results for an algorithm (and hyperparameters) running against a dataset in a HDF5 file.

    Args:
        dataset_name (str): The name of the dataset.
        count (int): The count of records.
        definition (Definition): The definition of the algorithm.
        query_arguments (Any): Additional arguments for the query.
        attrs (dict): Attributes to be stored in the file.
        results (list): Results to be stored.
        batch (bool): If True, the batch mode is activated.
    """
    filename = build_result_filepath(dataset_name, count, definition, query_arguments, batch, output_dir)
    directory, _ = os.path.split(filename)

    if not os.path.isdir(directory):
        os.makedirs(directory)

    with h5py.File(filename, "w") as f:
        attrs["result_file"] = filename
        if output_dir:
            attrs["output_dir"] = output_dir
            attrs["thread_suffix"] = get_thread_suffix()
        for k, v in attrs.items():
            f.attrs[k] = v
        times = f.create_dataset("times", (len(results),), "f")
        neighbors = f.create_dataset("neighbors", (len(results), count), "i")
        distances = f.create_dataset("distances", (len(results), count), "f")
        
        for i, (time, ds) in enumerate(results):
            times[i] = time
            neighbors[i] = [n for n, d in ds] + [-1] * (count - len(ds))
            distances[i] = [d for n, d in ds] + [float("inf")] * (count - len(ds))


def load_all_results(dataset: Optional[str] = None, 
                 count: Optional[int] = None, 
                 batch_mode: bool = False) -> Iterator[Tuple[dict, h5py.File]]:
    """
    Loads all the results from the HDF5 files in the specified path.

    Args:
        dataset (str, optional): The name of the dataset.
        count (int, optional): The count of records.
        batch_mode (bool, optional): If True, the batch mode is activated.

    Yields:
        tuple: A tuple containing properties as a dictionary and an h5py file object.
    """
    for root, _, files in os.walk(build_result_filepath(dataset, count)):
        for filename in files:
            if os.path.splitext(filename)[-1] != ".hdf5":
                continue
            try:
                with h5py.File(os.path.join(root, filename), "r+") as f:
                    properties = dict(f.attrs)
                    if batch_mode != properties["batch_mode"]:
                        continue
                    yield properties, f
            except Exception:
                print(f"Was unable to read {filename}")
                traceback.print_exc()


def get_unique_algorithms() -> Set[str]:
    """
    Retrieves unique algorithm names from the results.

    Returns:
        set: A set of unique algorithm names.
    """
    algorithms = set()
    for batch_mode in [False, True]:
        for properties, _ in load_all_results(batch_mode=batch_mode):
            algorithms.add(properties["algo"])
    return algorithms
