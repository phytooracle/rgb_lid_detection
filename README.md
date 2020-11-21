# RGB/FLIR Lettuce Detection
## Purpose
Detection of ground control point (GCP) lids in RGB imagery. 

## Inputs
Directory containing geoTIFFs. 

## Outputs
* CSV file containing:
    * Date
    * Plot
    * Genotype
    * Bounding box corner and center coordinates (EPSG:4326) 
    * Bounding area in m<sup>2</sup>. 

## Arguments and Flags
* **Positional Arguments:** 
    * **Single image or directory path:** 'dir' 
* **Required Arguments:**
    * **A .pth model file:** -m, --model
    * **Number of CPUs to use for multiprocessing:** -c, --cpu                  

* **Optional Arguments:**
    * **Output directory:** -o, --outdir, default=detect_out
    * **Prediction threshold:** -pt, --prediction_threshold, default=0.998
       
## Adapting the Script
                                        
## Running container using Singularity
**[Click here](https://hub.docker.com/repository/docker/phytooracle/rgb_lid_detection)** to access more information on this container. 

### Build the container:
`singularity build rgb_lid_detection.simg docker://phytooracle/rgb_lid_detection:latest`

### Print the help message:
`singularity run rgb_lid_detection.simg --help`

### Run the container:
'singularity run rgb_lid_detection.simg -m <model file path>.pth -c <number of CPUs> <single image or directory path>'
> **_NOTE:_**  To determine the number of CPUs available on your machine, run the following command: `lscpu`
