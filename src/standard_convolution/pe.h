#ifndef PE_H
#define PE_H
// Header guard: prevents file from being included multiple times during compilation.
// Generic PE that performs MAC (Multiply-Accumulate) operation.
// Used by all convolution types (standard, dilated, depthwise, grouped).

#include <systemc.h>

SC_MODULE(PE) {
    // Ports
    sc_in<bool> clk;
    sc_in<bool> rst;

    sc_in<int>  in_act;     // Activation coming from left
    sc_in<int>  in_psum;    // Partial sum coming from up

    sc_out<int> out_act;    // Activation passed to right
    sc_out<int> out_psum;   // New partial sum passed to down

    // Internal state
    int weight;

    int total_cycles  = 0;
    int active_cycles = 0;

    // MAC logic — triggered on every rising clock edge
    void compute() {
        if (rst.read() == true) {
            out_act.write(0);
            out_psum.write(0);
            total_cycles  = 0;
            active_cycles = 0;
        } else {
            total_cycles++;

            int act  = in_act.read();
            int psum = in_psum.read();

            if (act != 0)
                active_cycles++;

            // 1. Pass activation to the right
            out_act.write(act);

            // 2. MAC: psum_out = psum_in + (activation * weight)
            out_psum.write(psum + (act * weight));
        }
    }

    void set_weight(int w) { weight = w; }

    SC_HAS_PROCESS(PE);
    PE(sc_module_name name) : sc_module(name) {
        SC_METHOD(compute);
        sensitive << clk.pos();
        weight = 1;  // default; overwritten by load_weights()
    }
};

#endif // PE_H
