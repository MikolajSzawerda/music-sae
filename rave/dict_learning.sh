#!/bin/bash

./music-sae/rave/.venv/bin/python dictionary_learning.py ./activations/darbouka_decoder_7_30000 \
  darbouka_decoder_7_4_30000_50 \
  50 \
  dict_params.json \
  4 \
  1 \
  ./encoded/darbouka_decoder_7_30000/darbouka_decoder_7_4_30000_50 \
  ./weights/darbouka_decoder_7_4_30000_50_dictionary_weights.npy
