#ifndef DILATED_CONV_H
#define DILATED_CONV_H

#include <systemc.h>
#include <string>
#include <iostream>
#include "array.h"

// ============================================================================
// DilatedConv — Dilated 2-D convolution on a systolic array
//
// Template parameters:
//   IMG_H, IMG_W   — input image dimensions
//   K              — kernel size (K×K)
//   DILATION       — dilation rate (1 = standard convolution)
//
// The effective kernel footprint is:  EK = K + (K-1)*(DILATION-1)
// Output size:  OUT_H = IMG_H - EK + 1,  OUT_W = IMG_W - EK + 1
//
// Implementation strategy (the efficient one):
//   We never inflate the kernel.  For each output pixel (or, oc) we feed the
//   systolic array with the K×K samples taken from the input at stride =
//   DILATION.  The array's weights are loaded once with the original kernel.
//
//   The K×K systolic array computes one output pixel per K-cycle pass:
//     – Each row of the array corresponds to a kernel row.
//     – Activations for row kr are the K input values:
//         input[or + kr*DILATION][ oc + 0*DILATION ],
//         input[or + kr*DILATION][ oc + 1*DILATION ], ...
//       fed in successive cycles (skewed, as is standard for weight-stationary
//       systolic arrays).
//     – After 2*K-1 drain cycles the bottom-right psum_wire holds the result.
// ============================================================================

template <int IMG_H, int IMG_W, int K, int DILATION>
SC_MODULE(DilatedConv) {

    // Derived constants
    static const int EK    = K + (K - 1) * (DILATION - 1);  // effective kernel span
    static const int OUT_H = IMG_H - EK + 1;
    static const int OUT_W = IMG_W - EK + 1;
    static const int DRAIN = 2 * K - 1;  // cycles to drain one output through KxK array

    // Ports
    sc_in<bool> clk;
    sc_in<bool> rst;

    // Storage
    int image[IMG_H][IMG_W];
    int kernel_weights[K][K];
    int output[OUT_H][OUT_W];

    // The systolic array (K rows × K cols)
    SystolicArray<K, K>* sa;

    SC_HAS_PROCESS(DilatedConv);

    DilatedConv(sc_module_name name) : sc_module(name) {
        sa = new SystolicArray<K, K>("SA");
        sa->clk(clk);
        sa->rst(rst);
    }

    ~DilatedConv() {
        delete sa;
    }

    // ------------------------------------------------------------------
    // load_image / load_kernel — called before simulation starts
    // ------------------------------------------------------------------
    void load_image(const int img[IMG_H][IMG_W]) {
        for (int i = 0; i < IMG_H; i++)
            for (int j = 0; j < IMG_W; j++)
                image[i][j] = img[i][j];
    }

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
    // Why we iterate over kernel columns:
    //   In this array, activations flow left→right — so every column
    //   receives the SAME activation stream, just delayed.  Column j's
    //   weights w[kr][j] would multiply against the wrong image values
    //   if we tried to compute the full K×K dot-product in one pass.
    //
    //   Instead, we process one kernel column (kc) at a time:
    //     1. Load w[kr][kc] into column-0's PEs
    //     2. Skew-feed row kr with image[or + kr*D][oc + kc*D]
    //     3. After K cycles the psum chain produces:
    //          Σ_kr  w[kr][kc] * image[or + kr*D][oc + kc*D]
    //     4. Accumulate across all kc to get the full result.
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
                                int img_r = or_ + kr * DILATION;
                                int img_c = oc  + kc * DILATION;
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

#endif
