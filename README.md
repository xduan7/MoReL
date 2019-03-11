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

################################################################################
Getting features with dict_proxy method ... 
[Process 1] Training Utilization = 13.98% (33255 msec / 237942 msec)
[Process 0] Training Utilization = 30.09% (71477 msec / 237581 msec)
################################################################################
Getting features with mmap method ... 
[Process 1] Training Utilization = 12.47% (30373 msec / 243535 msec)
[Process 0] Training Utilization = 19.92% (48556 msec / 243809 msec)
################################################################################
Getting features with computing method ... 
[Process 1] Training Utilization = 8.26% (28501 msec / 344994 msec)
[Process 0] Training Utilization = 8.20% (28536 msec / 348205 msec)

################################################################################
Getting features with dict_proxy method ... 
[Process 1] Training Utilization = 14.01% (33520 msec / 239321 msec)
[Process 0] Training Utilization = 31.27% (74611 msec / 238637 msec)
################################################################################
Getting features with mmap method ... 
[Process 1] Training Utilization = 12.48% (29704 msec / 237931 msec)
[Process 0] Training Utilization = 22.49% (53514 msec / 237919 msec)
################################################################################
Getting features with computing method ... 
[Process 1] Training Utilization = 8.28% (28455 msec / 343739 msec)
[Process 0] Training Utilization = 8.15% (28068 msec / 344467 msec)

################################################################################
Getting features with computing method ... 
[Process 0] Training Utilization = 8.04% (28886 msec / 359262 msec)
[Process 1] Training Utilization = 8.21% (29549 msec / 359983 msec)
################################################################################
Getting features with mmap method ... 
[Process 1] Training Utilization = 13.65% (32364 msec / 237054 msec)
[Process 0] Training Utilization = 21.12% (50048 msec / 237015 msec)
################################################################################
Getting features with dict_proxy method ... 
[Process 1] Training Utilization = 14.39% (34152 msec / 237283 msec)
[Process 0] Training Utilization = 29.90% (70945 msec / 237245 msec)
