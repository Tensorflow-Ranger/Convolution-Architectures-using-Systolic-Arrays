#ifndef ARRAY_H
#define ARRAY_H
// No changes needed for depthwise convolution — the systolic array is
// channel-agnostic.  The caller (DepthwiseConv) reloads weights and re-drives
// inputs for each channel in turn.

#include <systemc.h>
#include <string>
#include "pe.h"

template <int ROWS, int COLS>
SC_MODULE(SystolicArray) {

    // Global ports
    sc_in<bool> clk;
    sc_in<bool> rst;

    // --- Wire grids ---
    // act_wire[i][0]    — array input  (left  edge)
    // act_wire[i][COLS] — array output (right edge)
    sc_signal<int> act_wire[ROWS][COLS + 1];

    // psum_wire[0][j]    — array input  (top    edge)
    // psum_wire[ROWS][j] — array output (bottom edge)
    sc_signal<int> psum_wire[ROWS + 1][COLS];

    // --- PE matrix ---
    PE* pe[ROWS][COLS];

    SC_CTOR(SystolicArray) {
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                std::string name = "PE_" + std::to_string(i) + "_" + std::to_string(j);
                pe[i][j] = new PE(name.c_str());

                pe[i][j]->clk(clk);
                pe[i][j]->rst(rst);

                // Horizontal: activations flow left → right
                pe[i][j]->in_act(act_wire[i][j]);
                pe[i][j]->out_act(act_wire[i][j + 1]);

                // Vertical: partial sums flow top → bottom
                pe[i][j]->in_psum(psum_wire[i][j]);
                pe[i][j]->out_psum(psum_wire[i + 1][j]);
            }
        }
    }

    void load_weights(int w_matrix[ROWS][COLS]) {
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++)
                pe[i][j]->set_weight(w_matrix[i][j]);
    }

    ~SystolicArray() {
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++)
                delete pe[i][j];
    }
};

#endif // ARRAY_H
