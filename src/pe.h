#ifndef PE_H
#define PE_H
//header guard, prevents file from being included multiple times during compilation
#include <systemc.h>

//defining the hardware components & signals
SC_MODULE(PE) {
    // Ports
    sc_in<bool> clk;
    sc_in<bool> rst;
    
    sc_in<int>  in_act;     // Activation coming from left
    sc_in<int>  in_psum;    // Partial sum coming from up
    
    sc_out<int> out_act;    // Activation passed to right
    sc_out<int> out_psum;   // New Partial sum passed to down

    // Internal State
    int weight; 

    // The MAC Logic / multiply accumulate
    //Run compute() on every rising edge of clock
    void compute() {
        /*If reset id active
        clear the outputs
        else */
        
        if (rst.read() == true) {
            out_act.write(0);
            out_psum.write(0);
        } else {
            int act = in_act.read();
            int psum = in_psum.read();
            
            // 1. Pass activation to the right
            out_act.write(act);
            
            // 2. MAC Operation: Psum_out = Psum_in + (Activation * Weight)
            out_psum.write(psum + (act * weight));
        }
    }

    // Constructor
    SC_HAS_PROCESS(PE); //tells that module has processes
    //name the hw instance smthg
    PE(sc_module_name name) : sc_module(name) {
        SC_METHOD(compute);//register the process
        sensitive << clk.pos(); 
        weight = 1; // Default weight for testing (will change this later with real kernel wights)
    }
};

#endif