#!/usr/bin/env bash

mkdir -p ".dataset"
# wget https://github.com/jayleicn/animeGAN/releases/download/data/anime-faces.tar.gz
mv anime-faces.tar.gz .dataset
tar xvf anime-faces.tar.gz
