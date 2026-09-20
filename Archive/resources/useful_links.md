# 🔗 Useful Links

A curated set of resources for Neurohack — background reading, tools, and inspiration for your BCI project.

## Contents
- [🔗 Useful Links](#-useful-links)
  - [Contents](#contents)
  - [General Learning Materials](#general-learning-materials)
  - [Tools \& Software](#tools--software)
    - [Python Libraries \& Tools](#python-libraries--tools)
    - [Other Software](#other-software)
    - [Open EEG Datasets](#open-eeg-datasets)
  - [BCI Game Design Inspiration](#bci-game-design-inspiration)
    - [Papers by Genre](#papers-by-genre)
    - [Other Inspiration](#other-inspiration)

---

## General Learning Materials
- **[AwesomeBCI Resources](https://github.com/NeuroTechX/awesome-bci)** — A really great compilation of BCI links/resources
- **[Previous year's submissions](https://github.com/SURGE-NeuroTech-Club/Neurohack-2025)** — Check out what past teams have built!

---

## Tools & Software

### Python Libraries & Tools
| Library/Tool | Use case |
|---|---|
| ⭐ **[SSVEP Toolbox](https://ssvep-toolbox.org/latest/)** ⭐ | Building and analyzing SSVEP-based BCIs — **highly recommended!** |
| **[MNE-Python](https://mne.tools/stable/index.html)** | EEG data analysis and visualization |
| **[BrainFlow](https://brainflow.org/)** | Cross-platform EEG data acquisition |
| **[SciKit-Learn](https://scikit-learn.org/)** | Machine learning for EEG classification |
| **[LabStreamingLayer (LSL)](https://labstreaminglayer.org/#/)** | Standard for real-time biosignal transmission |
| **[TensorFlow](https://www.tensorflow.org/)** / **[PyTorch](https://pytorch.org/)** | Deep learning for advanced EEG applications |
| **[PyGame](https://github.com/pygame/pygame)** | Visual and auditory library for BCI applications |
| **[Psychopy](https://psychopy.org/)** | Creating visual neuroscience experiments |
| **[MOABB](https://moabb.neurotechx.com/)** | Loads standard BCI datasets and benchmarks decoding pipelines against them |
| **[pyRiemann](https://pyriemann.readthedocs.io/)** | Riemannian-geometry classifiers — a go-to for high-accuracy EEG classification in BCI competitions |
| **[Braindecode](https://braindecode.org/)** | Deep learning toolbox built specifically for EEG/MEG decoding |
| **[NeuroKit2](https://neurokit2.readthedocs.io/)** | General-purpose neurophysiological signal processing (EEG, ECG, EDA, EMG, etc.) |

### Other Software
- **[OpenVibe](https://openvibe.inria.fr/)** — No-code (or Lua) real-time BCI development platform
- **[jsPsych](https://www.jspsych.org/latest/)** — JavaScript library for behavioral experiments
- **[OpenBCI GUI](https://docs.openbci.com/Software/OpenBCISoftware/GUIDocs/)** — Real-time EEG data streaming
- **[Unity](https://unity.com/)** — Building interactive BCI applications

### Open EEG Datasets
No headset yet, or want extra data to train/validate on? These are good places to look:
- **[OpenBCI: Publicly Available EEG Datasets](https://openbci.com/community/publicly-available-eeg-datasets/)** — Curated list of open EEG datasets across paradigms
- **[PhysioNet](https://physionet.org/)** — Hosts many public EEG datasets, incl. the Motor Movement/Imagery dataset
- **[MOABB dataset catalog](https://moabb.neurotechx.com/docs/datasets.html)** — Motor imagery, P300, and SSVEP datasets, loadable directly in Python

---

## BCI Game Design Inspiration

### Papers by Genre
- **[Tetris](https://home.isr.uc.pt/~gpires/papers/segah2011_submission_51_V2.pdf)**
- **[Shooter](https://pmc.ncbi.nlm.nih.gov/articles/PMC4804071/)**
- **[Shooter (deflect variant)](https://link.springer.com/chapter/10.1007/978-3-030-19591-5_6)**
- **[3D Environment](https://link.springer.com/article/10.1155/ASP.2005.3156)**
- **[Interactive Fiction](https://dl.acm.org/doi/pdf/10.1145/1864431.1864450)**
- **[Gomoku](https://pmc.ncbi.nlm.nih.gov/articles/PMC7956207/)**
- **[Battleship](https://pdfs.semanticscholar.org/836c/41575e62ac7533d1d4d79077876d651def48.pdf)**

> Not open access, but may be worth requesting via library access:
> - **[Checkers](https://ieeexplore.ieee.org/document/6943922)**
> - **[Racing](https://ieeexplore.ieee.org/document/6920562)**

### Other Inspiration
- **[br41n.io Spring School hackathon](https://www.br41n.io/Spring-School-2024)** — Past BCI hackathon projects
- **[itch.io Game Jams](https://itch.io/jams)** — See which game types/genres people rate as most fun
