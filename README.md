# Image Stitching and Panorama Generation

This project implements an image stitching pipeline for creating clean background composites and multi-image panoramas. The system uses SIFT feature extraction, descriptor matching with Lowe’s ratio test, RANSAC-based homography estimation, perspective warping, and mask-based blending to align overlapping images into a shared coordinate system.

## Project Overview

The project contains two main tasks:

### Task 1: Background Stitching with Moving Object Removal
Task 1 stitches two images of the same scene while reducing moving foreground objects. The pipeline first detects SIFT keypoints on grayscale images, matches feature descriptors, estimates a homography using RANSAC, and warps both images onto a common canvas. After alignment, the overlapping regions are compared to identify moving-object differences. Cleaner pixels from the alternate image are used to suppress foreground motion, followed by blending and cropping to produce the final stitched background.

### Task 2: Multi-Image Panorama Generation
Task 2 builds a panorama from multiple overlapping images. The code estimates pairwise homographies between image pairs, determines image connectivity using an overlap matrix, selects a reference image, composes transformations through the overlap graph, warps all connected images into one canvas, and blends them into a final panorama.

## Core Techniques

- SIFT feature detection and descriptor extraction
- Descriptor matching using Lowe’s ratio test
- RANSAC for robust homography estimation
- Homography-based perspective warping
- Canvas computation from transformed image corners
- Mask-based overlap detection
- Feather blending and seam smoothing
- Difference-based moving-object removal
- Overlap matrix generation for panorama connectivity
  
## Pipeline

Input Images  
    ↓  
Float Conversion and Normalization  
    ↓  
Grayscale Conversion  
    ↓  
SIFT Keypoint Detection + Descriptor Extraction  
    ↓  
Descriptor Matching with Ratio Test  
    ↓  
RANSAC Homography Estimation  
    ↓  
Canvas Computation  
    ↓  
Perspective Warping  
    ↓  
Valid-Region Mask Creation  
    ↓  
Overlap Detection  
    ↓  
Blending / Motion Removal  
    ↓  
Crop Black Borders  
    ↓  
Final Stitched Image / Panorama  
