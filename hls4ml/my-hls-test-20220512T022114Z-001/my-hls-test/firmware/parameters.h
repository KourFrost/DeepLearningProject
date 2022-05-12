#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_conv2d.h"
#include "nnet_utils/nnet_sepconv2d_stream.h"
 
//hls-fpga-machine-learning insert weights
#include "weights/w4.h"
#include "weights/b4.h"
#include "weights/w5.h"
#include "weights/b5.h"

//hls-fpga-machine-learning insert layer-config
// fc1
struct config4_mult : nnet::dense_config {
    static const unsigned n_in = tensor([ 0.6146, -1.1204,  0.3844,  0.5228, -0.2767, -0.0711, -0.9121, -0.9339,
         1.1184,  0.3993], grad_fn=<MulBackward0>);
    static const unsigned n_out = 512;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config4 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = N_INPUT_1_1;
    static const unsigned in_width = N_INPUT_2_1;
    static const unsigned n_chan = N_INPUT_3_1;
    static const unsigned filt_height = 1;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_4;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_4;
    static const unsigned out_width = OUT_WIDTH_4;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = N_INPUT_1_1;
    static const unsigned min_width = N_INPUT_2_1;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config4_mult mult_config;
};
const ap_uint<config4::filt_height * config4::filt_width> config4::pixels[] = {0};

// fc2
struct config5_mult : nnet::dense_config {
    static const unsigned n_in = 512;
    static const unsigned n_out = 10;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config5 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_4;
    static const unsigned in_width = OUT_WIDTH_4;
    static const unsigned n_chan = N_FILT_4;
    static const unsigned filt_height = 1;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_5;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_5;
    static const unsigned out_width = OUT_WIDTH_5;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = OUT_HEIGHT_4;
    static const unsigned min_width = OUT_WIDTH_4;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config5_mult mult_config;
};
const ap_uint<config5::filt_height * config5::filt_width> config5::pixels[] = {0};


#endif
