Molecule Representation Learning

TODO: 

1. Data download and featurization; 
    * Download PubChem data;
    * Transform into FP (ECFP4, ECFP6), SMILES, and graphs;
2. Models 
    * ~~Transformers~~;
    * ~~GGNN~~;
    * GCN;
3. Large scale training system
    * Shared data loader for multiple training threads (mmap?);
    * Process communication (HTTP?) for training control 
    (load and save, run on more GPUs, sync training processes);
4. Training and testing 
