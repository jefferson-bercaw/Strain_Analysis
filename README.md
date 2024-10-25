# Strain Analysis
#### Jefferson Bercaw
#### Last Updated: 10/25/2024
#### Contact: jrb187@duke.edu

## Overview:
This software is written to calculate the strain of cartilage due to exercise. The inputs are
.pcd (point cloud) files of the bones and cartilage surfaces both pre- and post-exercise. It registers
the post-exercise bone to the pre-exercise bone, and moves the post-exercise cartilage. After computing each
scan's cartilage thickness, it computes the strain map, and assigns it to the pre-exercise coordinate. It 
eliminates edge effects using an iterative, connectivity-based removal approach, and then finally shows the
strain map in a final plot.

## Installation:
This project was originally run on Python 3.11.9. To install the necessary packages, run the following command:
```bash
pip install -r requirements.txt --trusted-host=pypi.org
```

## Usage:
See example_usage.py for an example of how to use this software.

## Input Considerations:
The input .pcd files were exported from Geomagic Wrap, after remeshing to a target edge length of 0.6 mm.
This project was originally written for patella and patellar cartilage, so your specific application may be different. 
