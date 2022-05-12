//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t input1[N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1],
    result_t layer5_out[OUT_HEIGHT_5*OUT_WIDTH_5*N_FILT_5],
    unsigned short &const_size_in_1,
    unsigned short &const_size_out_1
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input1,layer5_out 
    #pragma HLS PIPELINE 

    const_size_in_1 = N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1;
    const_size_out_1 = OUT_HEIGHT_5*OUT_WIDTH_5*N_FILT_5;

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<model_default_t, 401408>(w4, "w4.txt");
        nnet::load_weights_from_txt<model_default_t, 512>(b4, "b4.txt");
        nnet::load_weights_from_txt<model_default_t, 5120>(w5, "w5.txt");
        nnet::load_weights_from_txt<model_default_t, 10>(b5, "b5.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer4_t layer4_out[OUT_HEIGHT_4*OUT_WIDTH_4*N_FILT_4];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::pointwise_conv_2d_cl<input_t, layer4_t, config4>(input1, layer4_out, w4, b4); // fc1

    nnet::pointwise_conv_2d_cl<layer4_t, result_t, config5>(layer4_out, layer5_out, w5, b5); // fc2

}
