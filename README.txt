README.TXT
There are codes of our proposed Token-Level Contrastive Learning with Modality-Aware Prompting for Multimodal
Intent Recognition (TCL-MAP).
- Code Framework
  - There are 5 important components in the framwork: configs for setting parameters, data for loading and 
    processing multimodal data, methods for constructing models and training, utils for computing metrics 
    and examples for defining script files.
- Dataset organization
  - MIntRec/MELD
    - train.tsv
    - dev.tsv
    - test.tsv
    - video_feats.pkl
    - audio_feats.pkl
  The internal structure of all files is consistent with the MIntRec dataset.
- Usage
  - You can configure your environment by using the following commands:
    - pip install -r requirements.txt
  - You can evaluate the performance of our proposed TCL-MAP on MIntRec and MELD-DA by using the following commands:
    - MIntRec: sh examples/run_TCL_MAP_MIntRec.sh
    - MELD-DA: sh examples/run_TCL_MAP_MELD.sh