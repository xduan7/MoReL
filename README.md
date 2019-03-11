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


################################################################################
Getting features with computing method ... 
[Process 0] Training Utilization = 8.74% (31910 msec / 365105 msec)
[Process 1] Training Utilization = 9.47% (34551 msec / 364999 msec)
################################################################################
Getting features with mmap method ... 
[Process 1] Training Utilization = 3.52% (68202 msec / 1940091 msec)
[Process 1] Hit 68237 times; Miss 62835 times
[Process 1] Feature hit ratio = 52.06%
[Process 0] Training Utilization = 3.41% (66087 msec / 1940464 msec)
[Process 0] Hit 62835 times; Miss 68237 times
[Process 0] Feature hit ratio = 47.94%
################################################################################
Getting features with dict_proxy method ... 
[Process 0] Training Utilization = 11.12% (53872 msec / 484658 msec)
[Process 0] Hit 65126 times; Miss 65946 times
[Process 0] Feature hit ratio = 49.69%
[Process 1] Training Utilization = 11.15% (54115 msec / 485209 msec)
[Process 1] Hit 65946 times; Miss 65126 times
[Process 1] Feature hit ratio = 50.31%
