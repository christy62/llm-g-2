# Treating Particle Track reconstruction as a Language Translation Problem

The [J-PARC muon g-2 experiment](https://g-2.kek.jp/overview/) is a high precision High Energy Physics experiment which will measure the anomalous magnetic moment and the electric dipole moment of the muon. 
This is measured by the tracks of decay positrons. The challenge in this experiment is to identify positron tracks and retrieve their momentum in a high pile-up scenario where multiple muon decay happens in a single time window.
Typically, the current algorithm uses the Hough Transforms and Kalman filters for this purpose which turn out to be computationally expensive.

In this Master's project, I explored an alternative approach by framing particle track reconstruction as a language translation problem, where the language of detectors is translated onto the language of physics. I have developed an LLM that generates a code segment to predict the hits back. My Master's thesis, datasets and LLM model can be requested from **christyelsa0602@gmail.com**.

# Repository Structure

- `fine_tune_llm.py` - Training script for the LLM model  
- `generate_decay_details.py` - Generates the decay momentum and the decay vertex of the particle
- `generate_hits.py` - Generates the hits after the physics processes
- `IMG` - This folder contains plots of few particle tracks

  These are the preliminary scripts used for data generation and model training.
  The inference code for this work is available on my [Colab Notebook](https://colab.research.google.com/drive/1c7w0GiQqgrdfyeTrrUDGuDNNodo4qTLU?usp=sharing).

  # Results

  Here are some of the events showing
 - `Fully matched tracks`
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/c6a9f9a1-80ee-4fac-ad55-d381ca619d95" width="250"/></td>
    <td><img src="https://github.com/user-attachments/assets/4f7bce9f-748f-495a-a3f9-00e94b97f937" width="250"/></td>
    <td><img src="https://github.com/user-attachments/assets/9857b331-cca6-431e-a8ae-f8ac2c1f7180" width="250"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/281bfed8-ce9b-4eb6-9d1a-09e952c8d06c" width="250"/></td>
    <td><img src="https://github.com/user-attachments/assets/945f4f95-9f60-4009-9b10-6210ea279b0f" width="250"/></td>
    <td><img src="https://github.com/user-attachments/assets/6562c150-debc-4e67-a8ac-2e2721b5b892" width="250"/></td>
  </tr>
</table>

  - `Partially matched tracks
 <table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/9a422fb7-83e0-4b89-864e-95e77b5241ce" width="250"/></td>
    <td><img src="https://github.com/user-attachments/assets/0f60a816-5108-4122-9655-e79c7a5e45ff" width="250"/></td>
    <td><img src="https://github.com/user-attachments/assets/d5793edd-b4cc-4d0c-b7f2-884f294f35a2" width="250"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/cc675fc5-c2cf-4574-92df-a414cdb15ad7" width="250"/></td>
    <td><img src="https://github.com/user-attachments/assets/07fe71c5-57a5-4e2b-a732-a4c63c61c876" width="250"/></td>
    <td><img src="https://github.com/user-attachments/assets/2b69d345-2c8f-4fc3-9d67-ddd8460b9ed4" width="250"/></td>
  </tr>
</table>




