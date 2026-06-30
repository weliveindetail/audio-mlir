#!/usr/bin/env bash
clang++ -O3 osc-low-pass.cpp -framework AudioToolbox -framework CoreAudio -o osc-low-pass
