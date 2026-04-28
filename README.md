# Treating particle track reconstruction as a language translation problem

The [J-PARC muon g-2 experiment](https://g-2.kek.jp/overview/) is a high precision High Energy Physics experiment which will measure the anomalous magnetic moment and the electric dipole moment of the muon. 
This is measured by the tracks of decay positrons. The challenge in this experiment is to identify positron tracks and retrieve their momentum in a high pile-up scenario where multiple muon decay happens in a single time window.
Typically, the current algorithm uses the Hough Transforms and Kalman filters for this purpose which turn out to be computationally expensive.
This work carried out as a part of my Master's project in which I have explored the use of Large Language Models, where the language of detectors is translated onto the language of physics. I have developed an LLM that generates a code segment to predict the hits back.
