#include <systemc.h>
#include "array.h"

int sc_main(int argc, char* argv[]) {
    sc_clock clk("clk", 10, SC_NS);
    sc_signal<bool> rst;

    SystolicArray<2, 2> my_array("Conv_Array");
    my_array.clk(clk);
    my_array.rst(rst);

    // --- NEW: Load Custom Weights ---
    int my_weights[2][2] = {
        {1, 2},
        {3, 4}
    };
    my_array.load_weights(my_weights);

    cout << "--- Starting 2x2 Array Simulation ---" << endl;
    
    // 1. Reset Phase
    rst.write(true);
    for(int i=0; i<2; i++) my_array.act_wire[i][0].write(0);  
    for(int j=0; j<2; j++) my_array.psum_wire[0][j].write(0); 
    sc_start(20, SC_NS); 
    
    // 2. Active Phase
    rst.write(false); 

    // Cycle 1
    my_array.act_wire[0][0].write(5);  // Row 0 gets a '5'
    my_array.psum_wire[0][0].write(0); 
    sc_start(10, SC_NS);

    // Cycle 2
    my_array.act_wire[0][0].write(0);  
    sc_start(10, SC_NS);

    // Cycle 3 (Flush it out)
    sc_start(10, SC_NS);

    // Print the counters from PE(0,0) ---
    cout << "Simulation Complete!" << endl;
    cout << "PE(0,0) Total Cycles:  " << my_array.pe[0][0]->total_cycles << endl;
    cout << "PE(0,0) Active Cycles: " << my_array.pe[0][0]->active_cycles << endl;
    
    // Calculate Utilization %
    float util = (float)my_array.pe[0][0]->active_cycles / my_array.pe[0][0]->total_cycles * 100;
    cout << "PE(0,0) Utilization:   " << util << "%" << endl;

    return 0;
}