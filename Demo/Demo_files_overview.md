# Demo files overview

## MS2LDA results files

1. ms2lda_viz.json [folder]
2. ms2lda.bin

## MGF file

1. Aligned feature list, created with feature extraction pipeline (e.g., in mzmine)


# Usage 

python MS2LDA_network_mappings_v1.0.py \
--mgf demo.mgf required=True \
--model ms2lda.bin required=True \
--viz ms2lda_viz.json required=True \
--outdir results_ms2lda_mapping required=True \
--threshold 0.002 type=float default=0.002 \
--mode both choices=["long", "wide", "both"] default="both"

## Example:

python MS2LDA_network_mappings_v1.0.py --mgf demo.mgf --model ms2lda.bin --viz ms2lda_viz.json --outdir results_ms2lda_mapping --threshold 0.002 type=float default=0.002 \--mode both 
