# Treating Particle Track reconstruction as a Language Translation Problem

The [J-PARC muon g-2 experiment](https://g-2.kek.jp/overview/) is a high precision High Energy Physics experiment which will measure the anomalous magnetic moment and the electric dipole moment of the muon. 
This is measured by the tracks of decay positrons. The challenge in this experiment is to identify positron tracks and retrieve their momentum in a high pile-up scenario where multiple muon decay happens in a single time window.
Typically, the current algorithm uses the Hough Transforms and Kalman filters for this purpose which turn out to be computationally expensive.

In this Master's project, I explore an alternative approach by framing particle track reconstruction as a language translation problem, where the language of detectors is translated onto the language of physics. I have developed an LLM that generates a code segment to predict the hits back.

# Repository Structure

- `fine_tune_llm.py` — Training script for the LLM model  
- `generate_decay_details.py` — Generates the decay momentum and the decay vertex of the particle

  These scripts are used for data generation, model training, and evaluation of track reconstruction performance.
