# Implementation to create common annotations

import glob
import argparse
import pandas as pd
from pathlib import Path

def ensure_path_exists(pathName):
    """
    Function to ensure that a path exists.
    The function would create a directory if that path is of a directory and the directory does not exists.

    Parameters
    ----------
    pathName   : str
                 Name of the path

    Returns
    -------
    pathName   : pathlib.Path
                 Name of the path
    """
    pathName = Path(pathName).resolve()
    if not pathName.exists():
      if not pathName.is_dir():
        pathName.mkdir(parents=True, exist_ok=False)

    return pathName

def create_common_annotations(pedestrianAnnotationsPath, sceneAnnotationsPath, outputDirectory):
    """
    Function to create a common annotations file.

    Parameters
    ----------
    pedestrianAnnotationsPath   : str
                                  Path to the pedestrian annotations file
    sceneAnnotationsPath        : str
                                  Path to the scene annotations file
    outputDirectory             : str
                                  Path to the output directtory

    Returns
    -------
    None
    """
    print("Creating common annotations file...")

    # Ensure that all the directories exist
    pedestrianAnnotationsPath = ensure_path_exists(pedestrianAnnotationsPath)
    sceneAnnotationsPath = ensure_path_exists(sceneAnnotationsPath)
    outputDirectory = ensure_path_exists(outputDirectory)

    # Get all the scene annotations paths
    sceneAnnotationsPath = str(sceneAnnotationsPath) + "/**/*.pkl"
    sceneAnnotationsPath = glob.glob(sceneAnnotationsPath, recursive=True)
    sceneAnnotationsPath = sorted(sceneAnnotationsPath)
    print("Total number of scene annotations found: {}".format(len(sceneAnnotationsPath)))

    # Read Pedestrian Annotations file
    with open(str(pedestrianAnnotationsPath), "rb") as pedestrianAnnotationsFile:
        pedestrianDatabase = pd.read_pickle(pedestrianAnnotationsFile)

    # Get all scene annotation values in one dictionary
    sceneDatabase = dict()
    # Append each Scene Annotations file to Common Annotations File
    for individualSceneAnnotationPath in sceneAnnotationsPath:
        with open(str(individualSceneAnnotationPath), "rb") as sceneAnnotationsFile:
            currentSceneDatabase = pd.read_pickle(sceneAnnotationsFile)
            sceneDatabase = dict(list(sceneDatabase.items()) + list(currentSceneDatabase.items()))
            
    # Perform Sanity Check
    assert (len(pedestrianDatabase.keys()) == len (sceneDatabase.keys())), "[ERROR] Number of keys in Pedestrian Database and Scene Database are different!"

    # Combine Pedestrian and Scene Annotations to create Common Annotations
    commonDatabase = pedestrianDatabase.copy()
    for individualKey in pedestrianDatabase.keys():
        if "vehicle_annotations" in sceneDatabase[individualKey].keys():
            commonDatabase[individualKey]["vehicle_annotations"] = sceneDatabase[individualKey]["vehicle_annotations"]
        else:
            print("[WARNING] Scene annotations not found for {}, creating empty dictionary...".format(individualKey))
            commonDatabase[individualKey]["vehicle_annotations"] = dict()

    # Save Common Annotations File as a Pickle File
    commonAnnotationsPath = outputDirectory / "overall_database.pkl"
    print("Path of the common annotations file: {}".format(commonAnnotationsPath))
    with open(str(commonAnnotationsPath), "w+") as commonAnnotationsFile:
        commonAnnotationsFile.write(str(commonDatabase))

    print("Succesfully created a common annotations file!!")

def main():
    """
    Entry point for the script.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="Script to create common annotations")
    parser.add_argument("-p", "--pedestrianAnnotationsPath", type=str, required=True, help="Path to the pedestrian annotations")
    parser.add_argument("-s", "--sceneAnnotationsPath", type=str, required=True, help="Path to the scene annotations")
    parser.add_argument("-o", "--outputDirectory", type=str, required=True, help="Path to the output directory")
    args =parser.parse_args()

    create_common_annotations(args.pedestrianAnnotationsPath, args.sceneAnnotationsPath, args.outputDirectory)

if __name__ == "__main__":
    main()
