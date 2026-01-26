# Nature Sense

Tools for recording and analyzing synchronized audio and video with acoustic beamforming.

## Overview

This repository contains two command-line tools for acoustic camera work:

- **record.py**: Records synchronized audio from a UMA-16 microphone array and video from a webcam
- **reprocess.py**: Reprocesses recorded sessions with acoustic beamforming to localize sound sources

## Hardware Requirements

- miniDSP UMA-16 microphone array (16-channel USB audio interface)
- Webcam (for video recording)

## Installation

### 1. Create Conda Environment

```bash
conda create -n nature_sense python=3.11
conda activate nature_sense
```

### 2. Install Dependencies

First install binary dependencies via conda, then pip install the rest:

```bash
conda install -c conda-forge numba
pip install -r requirements.txt
```
