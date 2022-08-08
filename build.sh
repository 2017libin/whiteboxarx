export PYTHONPATH="/home/chase/code/whiteboxarx/BoolCrypt:/home/chase/code/"

# Generating the affine layers
# sage -python speck.py --key 1918 1110 0908 0100 --block-size 32 --output-file speck32_64_affine_layers.sobj 

# Generating the implicit round functions
sage -python generate_wb.py --input-file speck32_64_affine_layers.sobj --irf-degree 2 --output-file speck32_64_irf.sobj

# Evaluating the implicit white-box implementation with Python
# sage -python eval_wb.py --input-file speck32_64_irf --plaintext 6574 694c --first-explicit-round "x = ((x >> 7) | (x << (WORD_SIZE - 7))); x = (x + y) & WORD_MASK;"
# sage -python eval_wb.py --input-file speck32_64_irf --plaintext 6574 694c --cancel-external-encodings 

# Evaluating the implicit white-box implementation with compiled C code
# sage -python export_wb.py --input-file speck32_64_irf.sobj --irf-degree 3 --first-explicit-round "x = ((x >> 7) | (x << (WORD_SIZE - 7))); x = (x + y) & WORD_MASK;"
# gcc white_box_arx.c -o white_box_arx m4ri/lib/libm4ri.a -I m4ri/include -lm > /dev/null 2>&1
# ./white_box_arx 6574 694c
