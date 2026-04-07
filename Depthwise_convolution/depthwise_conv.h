#ifndef DEPTHWISE_CONV_H
#define DEPTHWISE_CONV_H

#include <systemc.h>
#include <string>
#include <iostream>
#include "array.h"

// ============================================================================
// DepthwiseConv — Depthwise 2-D convolution on a systolic array
//
// Template parameters:
//   NUM_CHANNELS   — number of input (= output) channels
//   IMG_H, IMG_W   — spatial dimensions of each input channel
//   K              — kernel size (K×K)
//   DILATION       — dilation rate (1 = standard, >1 = dilated depthwise)
//
// Depthwise convolution differs from standard convolution in one key way:
//   • Each input channel is convolved with its OWN dedicated K×K kernel.
//   • There is NO cross-channel mixing (that is handled by a pointwise/1×1
//     conv in a full depthwise-separable stack).
//   • Input channels  = Output channels = NUM_CHANNELS.
//
// Hardware mapping:
//   We reuse the single K×K SystolicArray across all channels sequentially.
//   For each channel ch:
//     1. Load kernel_weights[ch] into the systolic array.
//     2. Run the same skewed-feed loop as DilatedConv.
//     3. Accumulate into output[ch][or][oc].
//
//   Effective kernel footprint: EK = K + (K-1)*(DILATION-1)
//   Output spatial size:  OUT_H = IMG_H - EK + 1
//                         OUT_W = IMG_W - EK + 1
// ============================================================================

template <int NUM_CHANNELS, int IMG_H, int IMG_W, int K, int DILATION>
SC_MODULE(DepthwiseConv) {

    // Derived constants
    static const int EK    = K + (K - 1) * (DILATION - 1);
    static const int OUT_H = IMG_H - EK + 1;
    static const int OUT_W = IMG_W - EK + 1;

    // Ports
    sc_in<bool> clk;
    sc_in<bool> rst;

    // -----------------------------------------------------------------------
    // Storage
    //   image          [ch][row][col]  — input feature map (all channels)
    //   kernel_weights [ch][kr][kc]    — one K×K kernel per channel
    //   output         [ch][or][oc]    — output feature map (all channels)
    // -----------------------------------------------------------------------
    int image         [NUM_CHANNELS][IMG_H][IMG_W];
    int kernel_weights[NUM_CHANNELS][K][K];
    int output        [NUM_CHANNELS][OUT_H][OUT_W];

    // One shared systolic array (K rows × K cols)
    SystolicArray<K, K>* sa;

    SC_HAS_PROCESS(DepthwiseConv);

    DepthwiseConv(sc_module_name name) : sc_module(name) {
        sa = new SystolicArray<K, K>("SA");
        sa->clk(clk);
        sa->rst(rst);
    }

    ~DepthwiseConv() { delete sa; }

    // ------------------------------------------------------------------
    // load_image — copy a multi-channel image into internal storage
    //   img[ch][row][col]
    // ------------------------------------------------------------------
    void load_image(const int img[NUM_CHANNELS][IMG_H][IMG_W]) {
        for (int ch = 0; ch < NUM_CHANNELS; ch++)
            for (int i = 0; i < IMG_H; i++)
                for (int j = 0; j < IMG_W; j++)
                    image[ch][i][j] = img[ch][i][j];
    }

    // ------------------------------------------------------------------
    // load_kernel — copy per-channel kernels into internal storage
    //   kern[ch][kr][kc]
    // ------------------------------------------------------------------
    void load_kernel(const int kern[NUM_CHANNELS][K][K]) {
        for (int ch = 0; ch < NUM_CHANNELS; ch++)
            for (int i = 0; i < K; i++)
                for (int j = 0; j < K; j++)
                    kernel_weights[ch][i][j] = kern[ch][i][j];
    }

    // ------------------------------------------------------------------
    // run() — compute depthwise convolution for all channels
    //
    // Outer loop: channels (ch = 0 .. NUM_CHANNELS-1)
    //   For each channel we reuse the single systolic array:
    //     1. Load this channel's K×K kernel.
    //     2. For every output spatial position (or_, oc):
    //        Loop over kernel columns (kc) — same column-by-column strategy
    //        as DilatedConv (see dilated_conv.h for the full rationale):
    //          a. Load kernel column kc into column-0 PEs.
    //          b. Reset the array (1 cycle).
    //          c. Skew-feed row kr with image[ch][or_+kr*D][oc+kc*D].
    //          d. After K cycles read psum_wire[K][0] and accumulate.
    //     3. Store accumulated value into output[ch][or_][oc].
    // ------------------------------------------------------------------
    void run(sc_signal<bool>& rst_sig) {
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {

            // Load this channel's weights into the systolic array once
            int w[K][K];
            for (int i = 0; i < K; i++)
                for (int j = 0; j < K; j++)
                    w[i][j] = kernel_weights[ch][i][j];
            sa->load_weights(w);

            for (int or_ = 0; or_ < OUT_H; or_++) {
                for (int oc = 0; oc < OUT_W; oc++) {
                    int total = 0;

                    for (int kc = 0; kc < K; kc++) {
                        // Override column-0 weights with kernel column kc
                        // (same column-by-column trick as DilatedConv)
                        for (int kr = 0; kr < K; kr++)
                            sa->pe[kr][0]->set_weight(kernel_weights[ch][kr][kc]);

                        // Reset the array (1 cycle)
                        rst_sig.write(true);
                        clear_inputs();
                        sc_start(10, SC_NS);

                        // Skew-feed activations through column-0
                        rst_sig.write(false);
                        for (int cycle = 0; cycle < K; cycle++) {
                            for (int kr = 0; kr < K; kr++) {
                                if (cycle == kr) {
                                    int img_r = or_ + kr * DILATION;
                                    int img_c = oc  + kc * DILATION;
                                    // Use this channel's image plane
                                    sa->act_wire[kr][0].write(image[ch][img_r][img_c]);
                                } else {
                                    sa->act_wire[kr][0].write(0);
                                }
                            }
                            for (int j = 0; j < K; j++)
                                sa->psum_wire[0][j].write(0);

                            sc_start(10, SC_NS);
                        }

                        // Read the accumulated psum from the bottom of column-0
                        total += sa->psum_wire[K][0].read();
                    }

                    output[ch][or_][oc] = total;
                }
            }
        }
    }

private:
    void clear_inputs() {
        for (int i = 0; i < K; i++)
            sa->act_wire[i][0].write(0);
        for (int j = 0; j < K; j++)
            sa->psum_wire[0][j].write(0);
    }
};

#endif // DEPTHWISE_CONV_H
