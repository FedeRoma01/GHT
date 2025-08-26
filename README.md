# GHT
Generalized Hough Transform implementation

#PROGRAMS
-serialeGHT.c: serial implementation of the GHT algorithm (single image)  
-parallelGHT.c: parallelized implementation based on the serial algorithm (single image)  
-serialeBatchGHT.c: serial batch implementation based on the serial algorithm (multiple images)  
-parallel3BatchGHT.c: parallelized batch implementation based on the first parallel program (multiple images)  

#UTILS
-batch_generating.py creates the dataset starting from the scene and its rotations that are in resources.zip  
-profiling_scripts.py must to be ran on the master, takes into account all the profilings done on each rank and extract the longest time with related function-time values  
-delete_profiling.sh needs to be used when working on the gcp to delete old profiling that could falsify newest recordings  
-profiling_gather.sh needs to be used when working on the gcp to gather all the rank profiling on the master that so is able to run profiling_scripts.py
