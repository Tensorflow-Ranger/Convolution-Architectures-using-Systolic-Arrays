#ifndef ARRAY_H
#define ARRAY_H

#include <systemc.h>
#include <string>
#include "pe.h"

template <int ROWS, int COLS>
SC_MODULE(SystolicArray) {
    // Global Ports
    sc_in<bool> clk;
    sc_in<bool> rst;

    // --- The Wire Grids ---
    // act_wire[i][0] is the Array Input (Left)
    // act_wire[i][COLS] is the Array Output (Right)
    sc_signal<int> act_wire[ROWS][COLS + 1];  

    // psum_wire[0][j] is the Array Input (Top)
    // psum_wire[ROWS][j] is the Array Output (Bottom)
    sc_signal<int> psum_wire[ROWS + 1][COLS]; 

    // --- The PE Matrix ---
    PE* pe[ROWS][COLS]; // We use pointers so we can instantiate them in a loop

    SC_CTOR(SystolicArray) {
        // The double loop to build the hardware
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                
                // 1. Generate a unique name for SystemC (e.g., "PE_0_0")
                std::string name = "PE_" + std::to_string(i) + "_" + std::to_string(j);
                
                // 2. Instantiate the PE
                pe[i][j] = new PE(name.c_str());

                // 3. Connect Clock & Reset
                pe[i][j]->clk(clk);
                pe[i][j]->rst(rst);

                // 4. Connect Horizontal Wires (Activations move Left -> Right)
                pe[i][j]->in_act(act_wire[i][j]);         // Input from West
                pe[i][j]->out_act(act_wire[i][j+1]);      // Output to East

                // 5. Connect Vertical Wires (Partial Sums move North -> South)
                pe[i][j]->in_psum(psum_wire[i][j]);       // Input from North
                pe[i][j]->out_psum(psum_wire[i+1][j]);    // Output to South
            }
        }
    }
    
    // Function to load a full matrix of weights into the array
    void load_weights(int w_matrix[ROWS][COLS]) {
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                pe[i][j]->set_weight(w_matrix[i][j]);
            }
        }
    }
    
    // Destructor to clean up memory
    ~SystolicArray() {
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                delete pe[i][j];
            }
        }
    }
};

#endif