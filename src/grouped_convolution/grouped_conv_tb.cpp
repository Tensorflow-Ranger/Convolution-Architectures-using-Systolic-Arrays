#include <systemc.h>
#include <iostream>
#include "grouped_conv.h"

// ---------- Software golden model (Hardware-Aware Grouped Conv) ----------
template <int IH, int IW, int K, int G>
void golden_grouped_conv(const int img[IH][IW], const int kern[K][K], int out[IH - K + 1][IW - K + 1]) {
    int row_chunk = K / G;
    int col_chunk = K / G;
    for (int r = 0; r < IH - K + 1; r++) {
        for (int c = 0; c < IW - K + 1; c++) {
            int total = 0;
            for (int kc = 0; kc < K; kc++) {
                int col_group = kc / col_chunk;
                for (int kr = 0; kr < K; kr++) {
                    int row_group = kr / row_chunk;
                    // Hardware only multiplies if row and column are in the same group
                    if (row_group == col_group) {
                        total += kern[kr][kc] * img[r + kr][c + kc];
                    }
                }
            }
            out[r][c] = total;
        }
    }
}

int sc_main(int argc, char* argv[]) {
    sc_clock clk("clk", 10, SC_NS);
    sc_signal<bool> rst;

    // Instantiate Grouped Convolution with 4x4 array (2 groups)
    const int IH = 6, IW = 6, K = 4, G = 2;
    GroupedConv<IH, IW, K, G> dut("GroupedConv_DUT");
    dut.clk(clk);
    dut.rst(rst);

    // --- VCD Tracing ---
    sc_trace_file *tf = sc_create_vcd_trace_file("grouped_waveforms");
    sc_trace(tf, clk, "clk");
    sc_trace(tf, rst, "rst");

    // Trace activations and psums from the array
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K + 1; j++) {
            std::string name = "act_" + std::to_string(i) + "_" + std::to_string(j);
            sc_trace(tf, dut.sa->act_wire[i][j], name.c_str());
        }
    }
    for (int i = 0; i < K + 1; i++) {
        for (int j = 0; j < K; j++) {
            std::string name = "psum_" + std::to_string(i) + "_" + std::to_string(j);
            sc_trace(tf, dut.sa->psum_wire[i][j], name.c_str());
        }
    }

    // Dummy data
    int image[IH][IW];
    for(int i=0; i<IH; i++) for(int j=0; j<IW; j++) image[i][j] = (i < IH/2) ? 2 : 5;

    int kernel[K][K];
    for(int i=0; i<K; i++) for(int j=0; j<K; j++) kernel[i][j] = 1;

    dut.load_image(image);
    dut.load_kernel(kernel);

    // Run simulation
    rst.write(true);
    sc_start(20, SC_NS);
    rst.write(false);

    dut.run(rst);

    // Results
    int golden[3][3];
    golden_grouped_conv<IH, IW, K, G>(image, kernel, golden);

    cout << "\n--- Validation ---" << endl;
    bool pass = true;
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            if(dut.output[i][j] != golden[i][j]) {
                cout << "Mismatch at [" << i << "][" << j << "]: Got " << dut.output[i][j] << ", Expected " << golden[i][j] << endl;
                pass = false;
            }
        }
    }

    if(pass) cout << "SUCCESS: Grouped convolution output matches golden model!" << endl;
    else cout << "FAILURE: Output mismatch." << endl;

    dut.print_utilization_report();

    // =========================================================
    //               POWER, AREA, AND TIME (PPA) ANALYSIS
    // =========================================================
    cout << "\n===============================================" << endl;
    cout << "           HARDWARE PPA ANALYSIS REPORT          " << endl;
    cout << "===============================================" << endl;

    // Constants for our Analytical Model
    const float CLOCK_PERIOD_NS = 10.0; 
    const float MAC_ENERGY_PJ = 5.0;     // Energy per active MAC (picoJoules)
    const float IDLE_ENERGY_PJ = 1.0;    // Leakage energy per cycle (picoJoules)
    const float PE_AREA_UM2 = 1500.0;    // Area per PE (micrometers squared)

    int global_active_cycles = 0;
    int global_total_cycles = 0; 

    // 1. Gather Data across the array (Using KxK since your kernel is K)
    for(int r = 0; r < K; r++) {
        for(int c = 0; c < K; c++) {
            // Access the PEs through your dut wrapper!
            global_active_cycles += dut.sa->pe[r][c]->active_cycles;
            if (dut.sa->pe[r][c]->total_cycles > global_total_cycles) {
                global_total_cycles = dut.sa->pe[r][c]->total_cycles;
            }
        }
    }

    // 2. TIME (Performance)
    float total_time_ns = global_total_cycles * CLOCK_PERIOD_NS;
    
    // 3. AREA
    int total_pes = K * K;
    float total_area = total_pes * PE_AREA_UM2;

    // 4. POWER (Energy Estimation)
    float dynamic_energy = global_active_cycles * MAC_ENERGY_PJ;
    float static_energy = (total_pes * global_total_cycles) * IDLE_ENERGY_PJ;
    float total_energy = dynamic_energy + static_energy;
    
    // Power = Energy / Time
    float average_power_mw = total_energy / total_time_ns; 

    // --- Print the Results ---
    cout << "[TIME] Total Latency:      " << total_time_ns << " ns (" << global_total_cycles << " cycles)" << endl;
    cout << "[AREA] Total Array Area:   " << total_area << " um^2 (" << total_pes << " PEs)" << endl;
    cout << "[POWER] Dynamic Energy:    " << dynamic_energy << " pJ" << endl;
    cout << "[POWER] Static Leakage:    " << static_energy << " pJ" << endl;
    cout << "[POWER] Total Energy:      " << total_energy << " pJ" << endl;
    cout << "[POWER] Avg Power Consump: " << average_power_mw << " mW" << endl;
    cout << "===============================================\n" << endl;

    sc_close_vcd_trace_file(tf);
    cout << "VCD file 'grouped_waveforms.vcd' generated." << endl;

    return pass ? 0 : 1;
}
