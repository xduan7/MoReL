Molecule Representation Learning

TODO: 

1. Data download and featurization; 
    * ~~Download PubChem data~~;
    * ~~Transform into FP (ECFP4, ECFP6), SMILES, and graphs~~;
    * Target from Dragon 7 (CIC5);
2. Models 
    * ~~Transformers~~;
    * ~~GGNN~~;
    * ~~GCN~~;
3. Large scale training system
    * ~~Shared data loader for multiple training threads (mmap or DictProxy)~~;
    * Process communication (HTTP?) for training control? 
    * (load and save, run on more GPUs, sync training processes);
4. Training and testing 
    * Training and testing functions
    * Learing rate changes (decay, early stop, etc.)
5. Final run
