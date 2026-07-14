# Bird Detection, Species Classification, and GIS Mapping with Co-DETR


The project uses **Co-DETR instance segmentation** to identify individual birds and generate high-resolution masks. These masks can then be processed using size, shape, color, spatial context, and additional classification models to estimate species and export survey results for GIS analysis.

## Project Goals

The primary goal is to reduce the amount of manual work required to:

- Detect birds in aerial survey imagery
- Generate accurate segmentation masks
- Distinguish birds from background objects
- Pre-classify birds using mask size, color, shape, and habitat context
- Classify selected bird species
- Convert image-space detections into geographic coordinates
- Export bird population survey results for GIS mapping and analysis

The system is designed to favor **high recall** during bird detection. False-positive detections can be reviewed and removed, while missed birds are much more difficult to recover without manually inspecting every survey image.
