#ifndef ARRAY_GROUPED_H
#define ARRAY_GROUPED_H

#include <systemc.h>
#include <string>
#include "pe.h" // Use your new PE!

template <int ROWS, int COLS>
SC_MODULE(SystolicArray_Grouped) {
    sc_in<bool> clk;
    sc_in<bool> rst;

    sc_signal<int> act_wire[ROWS][COLS + 1];  
    sc_signal<int> psum_wire[ROWS + 1][COLS]; 

    PE_Grouped* pe[ROWS][COLS]; 

    SC_CTOR(SystolicArray_Grouped) {
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                std::string name = "PE_" + std::to_string(i) + "_" + std::to_string(j);
                pe[i][j] = new PE_Grouped(name.c_str());

                pe[i][j]->clk(clk);
                pe[i][j]->rst(rst);

                pe[i][j]->in_act(act_wire[i][j]);         
                pe[i][j]->out_act(act_wire[i][j+1]);      

                pe[i][j]->in_psum(psum_wire[i][j]);       
                pe[i][j]->out_psum(psum_wire[i+1][j]);    
            }
        }
    }
    
    void load_weights(int w_matrix[ROWS][COLS]) {
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                pe[i][j]->set_weight(w_matrix[i][j]);
            }
        }
    }

    // --- NEW: Group Partitioning Logic ---
    void partition_groups(int num_groups) {
        int row_chunk = ROWS / num_groups;
        int col_chunk = COLS / num_groups;

        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                // If the PE falls in the block-diagonal chunk, it's active
                if ((i / row_chunk) == (j / col_chunk)) {
                    pe[i][j]->set_group_active(true);
                } else {
                    pe[i][j]->set_group_active(false); // Disable cross-talk!
                }
            }
        }
    }
    
    ~SystolicArray_Grouped() {
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                delete pe[i][j];
            }
        }
    }
};

#endif