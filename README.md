# Ego-Motion Aware Target Prediction (EMAP) Module for Robust Multi-Object Tracking

## Introduction
The EMAP (Ego-Motion Aware Target Prediction) module is a novel enhancement for detection-based multi-object tracking (DBT) systems, designed to mitigate the challenges posed by dynamic camera movements in tracking scenarios such as autonomous driving. This Kalman Filter-based prediction module significantly reduces identity switches and improves overall tracking performance by integrating camera motion and depth information into object motion models.

![EMAP Integration](emap_integration.jpg)  
*Figure 1: EMAP integration improves tracking performance in challenging scenarios.*

## Features
- **Integration with Existing MOT Algorithms:** Compatible with state-of-the-art trackers like OC-SORT, Deep OC-SORT, ByteTrack, and BoT-SORT.
- **Reduction in Identity Switches:** Demonstrates a significant reduction in identity switches, enhancing tracking continuity and accuracy.
- **Performance Enhancement:** Increases HOTA scores across various datasets and improves other metrics such as MOTA and IDF1.
- **Robust to Camera Motion:** Specifically adept at handling scenarios with significant ego-motion, such as autonomous vehicles.

## Datasets
- **KITTI MOT Dataset:** Used for benchmarking performance in real-world autonomous driving scenarios.
- **CARLA Simulation Dataset:** Utilized for controlled experimentations and to showcase EMAP's effectiveness in varied simulated autonomous driving environments.

## Getting Started
Clone the repository and install dependencies:
```bash
git clone https://github.com/noyzzz/EMAP.git
cd EMAP
pip install -r requirements.txt
```bash

