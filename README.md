# RGB Lid Detection
## Purpose
Detection of ground control point (GCP) lids in RGB imagery. Detection is accomplished using the **[Detecto](https://detecto.readthedocs.io/en/latest/)** Python package. A Faster R-CNN detection network outputs predictions, which are collected and output in a CSV file.

## Inputs
Single geoTIFF image or path to a directory containing multiple geoTIFF images. 

## Outputs
* CSV file containing the following columns:
    * image
    * center_x
    * center_y
    * min_x 
    * max_x
    * min_y
    * max_y

## Arguments and Flags
* **Positional Arguments:** 
    * **Single image or directory path:** 'dir' 
* **Required Arguments:**
    * **A .pth model file:** -m, --model
    * **Number of CPUs to use for multiprocessing:** -c, --cpu                  

* **Optional Arguments:**
    * **Output directory:** -o, --outdir, default=detect_out
    * **Prediction threshold:** -pt, --prediction_threshold, default=0.998
                                        
## Running container using Singularity
**[Click here](https://hub.docker.com/repository/docker/phytooracle/rgb_lid_detection)** to access this container. 

### Build the container:
```
singularity build rgb_lid_detection.simg docker://phytooracle/rgb_lid_detection:latest
```

### Print the help message:
```
singularity run rgb_lid_detection.simg --help
```

### Run the container:
```
singularity run rgb_lid_detection.simg -m <model file path>.pth -c <number of CPUs> <single image or directory path>
```
> **_NOTE:_**  To determine the number of CPUs available on your machine, run the following command: `lscpu`
