#ifndef STANDARD_CONV_H
#define STANDARD_CONV_H

#include <systemc.h>
#include <string>
#include <iostream>
#include "array.h"

// ============================================================================
// StandardConv — Standard 2-D convolution on a systolic array
//
// Template parameters:
//   IMG_H, IMG_W   — input image dimensions
//   K              — kernel size (K×K)
//
// Standard convolution is the baseline convolution operation:
//   out[r][c] = Σ_{kr,kc}  kernel[kr][kc] * image[r + kr][c + kc]
//
// This is equivalent to dilated convolution with DILATION=1.
//
// Output size:  OUT_H = IMG_H - K + 1,  OUT_W = IMG_W - K + 1
//
// Implementation strategy:
//   We reuse the efficient column-by-column feeding approach to avoid
//   any inflation of the kernel. For each output pixel (or, oc):
//     1. For each kernel column kc:
//        a. Load w[kr][kc] into the column-0 PEs
//        b. Skew-feed row kr with image[or + kr][oc + kc]
//        c. After K cycles accumulate the result
//     2. Sum contributions from all kernel columns to get final output.
//
//   The K×K systolic array operates in weight-stationary mode, where
//   weights remain fixed at their PEs throughout computation.
// ============================================================================

template <int IMG_H, int IMG_W, int K>
SC_MODULE(StandardConv) {

    // Derived constants
    static const int OUT_H = IMG_H - K + 1;
    static const int OUT_W = IMG_W - K + 1;

    // Ports
    sc_in<bool> clk;
    sc_in<bool> rst;

    // Storage
    int image[IMG_H][IMG_W];
    int kernel_weights[K][K];
    int output[OUT_H][OUT_W];

    // The systolic array (K rows × K cols)
    SystolicArray<K, K>* sa;

    SC_HAS_PROCESS(StandardConv);

    StandardConv(sc_module_name name) : sc_module(name) {
        sa = new SystolicArray<K, K>("SA");
        sa->clk(clk);
        sa->rst(rst);
    }

    ~StandardConv() {
        delete sa;
    }

    // ------------------------------------------------------------------
    // load_image — copy image into internal storage
    //   img[row][col]
    // ------------------------------------------------------------------
    void load_image(const int img[IMG_H][IMG_W]) {
        for (int i = 0; i < IMG_H; i++)
            for (int j = 0; j < IMG_W; j++)
                image[i][j] = img[i][j];
    }

    // ------------------------------------------------------------------
    // load_kernel — copy kernel into internal storage and into the array
    //   kern[kr][kc]
    // ------------------------------------------------------------------
    void load_kernel(const int kern[K][K]) {
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++)
                kernel_weights[i][j] = kern[i][j];

        // Load weights into the systolic array PEs
        int w[K][K];
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++)
                w[i][j] = kern[i][j];
        sa->load_weights(w);
    }

    // ------------------------------------------------------------------
    // run() — drive the array to compute every output pixel
    //
    // For each output spatial position (or_, oc):
    //   For each kernel column (kc):
    //     1. Load kernel column kc weights into column-0 PEs
    //     2. Reset the array (1 cycle)
    //     3. Skew-feed activations: row kr receives image[or_ + kr][oc + kc]
    //     4. After K cycles accumulate the result contribution
    //   Sum all kernel column contributions to get final output value
    // ------------------------------------------------------------------
    void run(sc_signal<bool>& rst_sig) {
        for (int or_ = 0; or_ < OUT_H; or_++) {
            for (int oc = 0; oc < OUT_W; oc++) {
                int total = 0;

                for (int kc = 0; kc < K; kc++) {
                    // Load kernel column kc into column 0 of the array
                    for (int kr = 0; kr < K; kr++)
                        sa->pe[kr][0]->set_weight(kernel_weights[kr][kc]);

                    // Reset the array (1 cycle)
                    rst_sig.write(true);
                    clear_inputs();
                    sc_start(10, SC_NS);

                    // Feed activations with row skew through column 0
                    rst_sig.write(false);
                    for (int cycle = 0; cycle < K; cycle++) {
                        for (int kr = 0; kr < K; kr++) {
                            if (cycle == kr) {
                                int img_r = or_ + kr;
                                int img_c = oc  + kc;
                                sa->act_wire[kr][0].write(image[img_r][img_c]);
                            } else {
                                sa->act_wire[kr][0].write(0);
                            }
                        }
                        for (int j = 0; j < K; j++)
                            sa->psum_wire[0][j].write(0);

                        sc_start(10, SC_NS);
                    }

                    // After K cycles the psum chain has fully propagated
                    total += sa->psum_wire[K][0].read();
                }

                output[or_][oc] = total;
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

#endif // STANDARD_CONV_H
