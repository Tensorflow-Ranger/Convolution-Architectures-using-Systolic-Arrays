#include <systemc.h>
#include "array.h"

int sc_main(int argc, char* argv[]) {
    // 1. Setup Clock and Reset
    sc_clock clk("clk", 10, SC_NS);
    sc_signal<bool> rst;

    // 2. Instantiate the 2x2 Array
    SystolicArray<2, 2> my_array("Conv_Array");
    my_array.clk(clk);
    my_array.rst(rst);


    sc_trace_file *wf = sc_create_vcd_trace_file("array_wave");

    sc_trace(wf, clk, "clk");
    sc_trace(wf, rst, "rst");

    for(int i=0;i<2;i++)
        for(int j=0;j<=2;j++)
            sc_trace(wf, my_array.act_wire[i][j], "act");

    for(int i=0;i<=2;i++)
        for(int j=0;j<2;j++)
            sc_trace(wf, my_array.psum_wire[i][j], "psum");
    
            
    // 3. Run the Simulation
    cout << "--- Starting 2x2 Systolic Array Simulation ---" << endl;
    
    // Reset the array
    rst.write(true);
    
    // Zero out the inputs at the edges
    for(int i=0; i<2; i++) my_array.act_wire[i][0].write(0);  // Left edge
    for(int j=0; j<2; j++) my_array.psum_wire[0][j].write(0); // Top edge
    
    sc_start(10, SC_NS); 
    rst.write(false); // Release reset

    // ---------------------------------------------------------
    // CYCLE 1: Feed data to PE(0,0)
    // ---------------------------------------------------------
    my_array.act_wire[0][0].write(2);  // Pixel input at Row 0
    my_array.psum_wire[0][0].write(0); // Initial partial sum at Col 0
    sc_start(10, SC_NS);
    cout << "Cycle 1 complete." << endl;

    // ---------------------------------------------------------
    // CYCLE 2: Data moves deeper into the array
    // ---------------------------------------------------------
    // PE(0,0) output is now sitting on act_wire[0][1] and psum_wire[1][0]
    // Feed new data into the edges
    my_array.act_wire[0][0].write(3);  
    my_array.act_wire[1][0].write(4);  // Pixel input at Row 1
    sc_start(10, SC_NS);
    
    // Let's print the wires inside the array to prove it workes
    cout << "Cycle 2 -> PE(0,0) passed Activation right: " << my_array.act_wire[0][1].read() << endl;
    cout << "Cycle 2 -> PE(0,0) passed Psum down:      " << my_array.psum_wire[1][0].read() << endl;

    // ---------------------------------------------------------
    // CYCLE 3: Wait for PE(1,1) to finish
    // ---------------------------------------------------------
    my_array.act_wire[0][0].write(0);
    my_array.act_wire[1][0].write(0);
    sc_start(10, SC_NS);
    
    // Read the FINAL output at the bottom of Column 0
    cout << "Cycle 3 -> Final Psum out of Col 0 (Bottom): " << my_array.psum_wire[2][0].read() << endl;

    
    sc_close_vcd_trace_file(wf);

    cout << "--- Simulation Complete ---" << endl;
    
    return 0;
}