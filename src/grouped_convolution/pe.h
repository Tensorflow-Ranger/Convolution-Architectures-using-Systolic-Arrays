#ifndef PE_GROUPED_H
#define PE_GROUPED_H

#include <systemc.h>

SC_MODULE(PE_Grouped) {
    sc_in<bool> clk;
    sc_in<bool> rst;
    
    sc_in<int>  in_act;     
    sc_in<int>  in_psum;    
    
    sc_out<int> out_act;    
    sc_out<int> out_psum;   

    int weight; 
    
    // --- NEW: Grouped Conv Logic ---
    bool is_active_for_group; // If false, this PE is power-gated

    int total_cycles = 0;
    int active_cycles = 0;

    void compute() {
        if (rst.read() == true) {
            out_act.write(0);
            out_psum.write(0);
            total_cycles = 0;
            active_cycles = 0;
        } else {
            total_cycles++;

            int act = in_act.read();
            int psum = in_psum.read();

            // 1. Pass activation to the right (Data must still flow!)
            out_act.write(act);

            // 2. MAC Operation or Power Gated Bypass
            if (is_active_for_group) {
                if (act != 0) active_cycles++; // Only count if actually working
                out_psum.write(psum + (act * weight)); // Do MAC
            } else {
                // POWER GATED: Skip MAC, just pass the psum down safely
                out_psum.write(psum);
            }
        }
    }

    void set_weight(int w) { weight = w; }
    
    // Function to enable/disable the PE based on Array Partitioning
    void set_group_active(bool active) { is_active_for_group = active; }

    SC_HAS_PROCESS(PE_Grouped); 
    PE_Grouped(sc_module_name name) : sc_module(name) {
        SC_METHOD(compute);
        sensitive << clk.pos(); 
        weight = 1; 
        is_active_for_group = true; // Default to standard behavior
    }
};

#endif