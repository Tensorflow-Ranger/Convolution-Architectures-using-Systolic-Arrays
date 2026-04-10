#ifndef GROUPED_CONV_H
#define GROUPED_CONV_H

#include <systemc.h>
#include <string>
#include <iostream>
#include "array.h"

// ============================================================================
// GroupedConv — Grouped convolution on a systolic array
//
// Template parameters:
//   IMG_H, IMG_W   — input image dimensions
//   K              — kernel size (K×K)
//   NUM_GROUPS     — number of convolution groups
//
// The Grouped Convolution divides the input channels and filters into groups.
// In this systolic array implementation, we partition the array itself into
// block-diagonal segments. PEs outside these segments are "power-gated".
// ============================================================================

template <int IMG_H, int IMG_W, int K, int NUM_GROUPS>
SC_MODULE(GroupedConv) {

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
    SystolicArray_Grouped<K, K>* sa;

    SC_HAS_PROCESS(GroupedConv);

    GroupedConv(sc_module_name name) : sc_module(name) {
        sa = new SystolicArray_Grouped<K, K>("SA");
        sa->clk(clk);
        sa->rst(rst);
        
        // Partition the groups
        sa->partition_groups(NUM_GROUPS);
    }

    ~GroupedConv() {
        delete sa;
    }

    void load_image(const int img[IMG_H][IMG_W]) {
        for (int i = 0; i < IMG_H; i++)
            for (int j = 0; j < IMG_W; j++)
                image[i][j] = img[i][j];
    }

    void load_kernel(const int kern[K][K]) {
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++)
                kernel_weights[i][j] = kern[i][j];

        sa->load_weights(kernel_weights);
    }

    void run(sc_signal<bool>& rst_sig) {
        cout << "--- Starting Grouped Conv Simulation (" << NUM_GROUPS << " Groups) ---" << endl;
        
        for (int or_ = 0; or_ < OUT_H; or_++) {
            for (int oc = 0; oc < OUT_W; oc++) {
                int total = 0;

                for (int kc = 0; kc < K; kc++) {
                    // Update weights for this kernel column (use column 'kc' of the array)
                    for (int kr = 0; kr < K; kr++)
                        sa->pe[kr][kc]->set_weight(kernel_weights[kr][kc]);

                    // Reset phase
                    rst_sig.write(true);
                    clear_inputs();
                    sc_start(10, SC_NS);

                    // Stream activations
                    rst_sig.write(false);
                    // Run for enough cycles to let data reach column 'kc' and psum reach bottom
                    // Latency = kc (to reach col) + K (to reach bottom)
                    int total_cycles = K + kc + 2; // Extra buffer
                    int val = 0;
                    for (int cycle = 0; cycle < total_cycles; cycle++) {
                        for (int kr = 0; kr < K; kr++) {
                            if (cycle == kr) {
                                sa->act_wire[kr][0].write(image[or_ + kr][oc + kc]);
                            } else {
                                sa->act_wire[kr][0].write(0);
                            }
                        }
                        for (int j = 0; j < K; j++)
                            sa->psum_wire[0][j].write(0);

                        sc_start(10, SC_NS);
                        
                        // Capture the highest value reached (peak detector)
                        int current_psum = sa->psum_wire[K][kc].read();
                        if (current_psum > val) val = current_psum;
                    }
                    total += val;
                }

                output[or_][oc] = total;
            }
        }
    }

    void print_utilization_report() {
        cout << "\n--- OPTIMIZATION REPORT (Grouped Conv) ---" << endl;
        for (int i = 0; i < K; i+= (K/NUM_GROUPS)) {
            for (int j = 0; j < K; j+= (K/NUM_GROUPS)) {
                float util = (float)sa->pe[i][j]->active_cycles / sa->pe[i][j]->total_cycles * 100;
                bool is_active = sa->pe[i][j]->is_active_for_group;
                std::string status = is_active ? "[ACTIVE]" : "[GATED ]";
                cout << status << " PE(" << i << "," << j << ") Utilization: " << util << "%" << endl;
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
