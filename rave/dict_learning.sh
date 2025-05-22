#!/bin/bash

/home/mkielbus/music-sae/rave/.venv/bin/python dictionary_learning.py ./activations/darbouka_decoder_7_30000 \
  darbouka_decoder_7_30000_50 \
  50 \
  dict_params.json \
  4 \
  70000 \
  ./encoded/darbouka_decoder_7_30000/darbouka_decoder_4_50 \
  ./weights/darbouka_decoder_7_4_50_dictionary_weights.npy
