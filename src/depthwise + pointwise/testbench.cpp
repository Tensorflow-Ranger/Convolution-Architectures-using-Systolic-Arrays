#include "design.h"
#include <cstdio>
#include <vector>

SC_MODULE(Testbench) {
    sc_out<bool> clk;
    sc_out<bool> reset;
    sc_out<bool> enable[NUM_PE];

    sc_out<sc_int<DATA_WIDTH>> data[NUM_PE];
    sc_out<sc_int<DATA_WIDTH>> weight;

    sc_in<sc_int<ACC_WIDTH>> result[NUM_PE];
    sc_in<bool> done[NUM_PE];

    SC_CTOR(Testbench) {
        SC_THREAD(run);
    }

    void tick() {
        clk.write(1); wait(5, SC_NS);
        clk.write(0); wait(5, SC_NS);
    }

    void run() {
        clk.write(0);
        reset.write(1);
        weight.write(0);
        for (int i = 0; i < NUM_PE; i++) {
            data[i].write(0);
            enable[i].write(true);
        }
        tick(); tick();
        reset.write(0);
        tick();

        printf("\nSimulation Started for Depthwise + Pointwise!\n");

        // ---------------------------------------------------------
        // Synthetic Input Data
        // ---------------------------------------------------------
        // image_in: 3x3 spatial, 3 channels (Cin = 3)
        int image_in[C_IN][K_SIZE * K_SIZE] = {
            {1, 2, 0, 1, 1, 1, 0, 2, 1}, // Ch 0
            {0, 1, 1, 2, 0, 2, 1, 1, 0}, // Ch 1
            {2, 0, 1, 1, 2, 0, 0, 1, 2}  // Ch 2
        };

        // Depthwise Weights (one 3x3 filter per channel)
        int dw_weight[C_IN][K_SIZE * K_SIZE] = {
            {1, 0, -1, 1, 0, -1, 1, 0, -1}, // Ch 0
            {0, 1, 0, 1, 1, 1, 0, 1, 0},    // Ch 1
            {-1, -1, -1, 0, 0, 0, 1, 1, 1}  // Ch 2
        };

        // Pointwise Weights (C_OUT x C_IN = 4 x 3)
        int pw_weight[C_OUT][C_IN] = {
            {1, 2, 1},  // Output Ch 0
            {0, 1, -1}, // Output Ch 1
            {0, 0, 1},  // Output Ch 2
            {1, -1, 0}  // Output Ch 3
        };

        // Software Golden Model
        int golden_dw_out[C_IN] = {0};
        for (int c = 0; c < C_IN; c++) {
            for (int k = 0; k < K_SIZE * K_SIZE; k++) {
                golden_dw_out[c] += image_in[c][k] * dw_weight[c][k];
            }
        }

        int golden_pw_out[C_OUT] = {0};
        for (int oc = 0; oc < C_OUT; oc++) {
            for (int ic = 0; ic < C_IN; ic++) { // BUG was here, oc < C_OUT condition nested incorrectly
                golden_pw_out[oc] += golden_dw_out[ic] * pw_weight[oc][ic];
            }
        }

        // ---------------------------------------------------------
        // HARDWARE SIMULATION
        // ---------------------------------------------------------

        // --- STAGE 1: DEPTHWISE CONVOLUTION ---
        printf("\n--- STAGE 1: DEPTHWISE --- \n");
        // depthwise output array initialization
        int hw_dw_out[C_IN] = {0};

        // For depthwise, each channel processed independently over the shared hardware.
        // By doing it entirely one-by-one, we avoid data racing from continuous weight tracking.
        for (int c = 0; c < C_IN; c++) {
            // First stop driving PE values to ensure clear boundary for channels
            for(int p = 0; p < NUM_PE; p++) {
                data[p].write(0);
            }
            weight.write(0);
            
            // Give a complete reset sequence per channel iteration
            clk.write(0);
            reset.write(1);
            tick();
            reset.write(0);
            tick();
            
            for (int k = 0; k < K_SIZE * K_SIZE; k++) {
                data[c].write(image_in[c][k]); 
                weight.write(dw_weight[c][k]);
                tick();
            }
            
            // wait for calculation to be done
            for(int wt = 0; wt < 2; wt++) tick();
            hw_dw_out[c] = result[c].read();
        }

        for (int c = 0; c < C_IN; c++) {
            printf("  DW_Out[Ch %d] -> HW: %2d | Golden: %2d\n", c, hw_dw_out[c], golden_dw_out[c]);
        }

        // --- STAGE 2: POINTWISE CONVOLUTION ---
        printf("\n--- STAGE 2: POINTWISE --- \n");
        int hw_pw_out[C_OUT] = {0};

        // Reset test states
        for (int ic = 0; ic < NUM_PE; ic++) {
             clk.write(0);
             reset.write(1);
             tick();
             reset.write(0);
             tick();
        }

        // 1x1 convolution over channels
        for (int ic = 0; ic < C_IN; ic++) {
            weight.write(hw_dw_out[ic]); // Must use the hardware computed dw_out instead of golden_dw_out
            for (int oc = 0; oc < C_OUT; oc++) {
                data[oc].write(pw_weight[oc][ic]);
            }
            tick();
        }
        
        // Push dummies if K_SIZE * K_SIZE wasn't fully reached to satisfy PE loop
        for (int fill = C_IN; fill < K_SIZE * K_SIZE; fill++) {
            weight.write(0);
            for (int oc=0; oc < C_OUT; oc++) data[oc].write(0);
            tick();
        }
        
        for(int wt=0; wt<3; wt++) tick();
        
        bool pass = true;
        for (int oc = 0; oc < C_OUT; oc++) {
            hw_pw_out[oc] = result[oc].read();
            printf("  PW_Out[Ch %d] -> HW: %2d | Golden: %2d\n", oc, hw_pw_out[oc], golden_pw_out[oc]);
            if (hw_pw_out[oc] != golden_pw_out[oc]) pass = false;
        }

        printf("\n=> FUNCTIONAL VALIDATION: %s\n", pass ? "SUCCESS" : "FAIL");

        int active_standard = 0;
        int active_depthwise = 0;
        int active_pointwise = 0;
        int cycles_standard = 0;
        int cycles_depthwise = 0;
        int cycles_pointwise = 0;

        // Constants used: K_SIZE = 3, C_IN = 3, C_OUT = 4
        int k_sq = K_SIZE * K_SIZE;

        // 1. Standard Convolution
        // Cycles = k_sq * C_IN * C_OUT ; all PEs active
        cycles_standard = k_sq * C_IN * C_OUT;
        active_standard = NUM_PE * cycles_standard; 

        // 2. Depthwise Separable
        // Step A: Depthwise = k_sq * C_IN
        cycles_depthwise = k_sq * C_IN;
        active_depthwise = NUM_PE * cycles_depthwise;

        // Step B: Pointwise = C_IN * C_OUT  (1x1 convolution across channels)
        cycles_pointwise = C_IN * C_OUT;
        active_pointwise = NUM_PE * cycles_pointwise;

        int total_separable_cycles = cycles_depthwise + cycles_pointwise;
        int total_separable_active = active_depthwise + active_pointwise;

        printf("\n=============================================\n");
        printf("        PPA COMPARISON REPORT\n");
        printf("=============================================\n");

        int latency_ns_standard = cycles_standard * 10;
        int latency_ns_separable = total_separable_cycles * 10;

        int physical_area = NUM_PE * 1500;

        float energy_standard = active_standard * 2.0;         // ~2.0 pJ per MAC
        float energy_separable = total_separable_active * 2.0; // ~2.0 pJ per MAC

        float leakage_standard = cycles_standard * NUM_PE * 5.0; // static power
        float leakage_separable = total_separable_cycles * NUM_PE * 5.0;

        float total_metric_std = energy_standard + leakage_standard;
        float total_metric_sep = energy_separable + leakage_separable;

        printf("[AREA ] Physical Area             : %d um^2 (Constant SystemC design)\n", physical_area);
        printf("[PARAM] Standard Parameters       : %d (K*K*Cin*Cout)\n", (k_sq * C_IN * C_OUT));
        printf("[PARAM] Separable Parameters      : %d (K*K*Cin + Cin*Cout)\n", (k_sq * C_IN) + (C_IN * C_OUT));

        printf("\n--- LATENCY ---\n");
        printf("Standard Convolution Latency      : %d ns (%d cycles)\n", latency_ns_standard, cycles_standard);
        printf("Depthwise+Pointwise Latency       : %d ns (%d cycles)\n", latency_ns_separable, total_separable_cycles);

        printf("\n--- ENERGY ---\n");
        printf("Standard Convolution Energy       : %.2f pJ\n", total_metric_std);
        printf("Depthwise+Pointwise Energy        : %.2f pJ\n", total_metric_sep);

        float param_reduction = (float)((k_sq * C_IN * C_OUT) - ((k_sq * C_IN) + (C_IN * C_OUT))) / (k_sq * C_IN * C_OUT) * 100.0;
        float cycle_saving = (float)(cycles_standard - total_separable_cycles) / cycles_standard * 100.0;
        float power_saving = (float)(total_metric_std - total_metric_sep) / total_metric_std * 100.0;

        printf("\n>>> PARAMETER REDUCTION   : %.2f%%\n", param_reduction);
        printf(">>> COMPUTE/CYCLE SAVING  : %.2f%%\n", cycle_saving);
        printf(">>> TOTAL ENERGY SAVING   : %.2f%%\n", power_saving);

        printf("\n(Note: Cost reduces from K^2 * Cin * Cout to K^2 * Cin + Cin * Cout.)\n");
        printf("=============================================\n\n");

        sc_stop();
    }
};

int sc_main(int argc, char* argv[]) {
    sc_signal<bool> clk_s, reset_s;
    sc_signal<bool> enable_s[NUM_PE];

    sc_signal<sc_int<DATA_WIDTH>> data_s[NUM_PE];
    sc_signal<sc_int<DATA_WIDTH>> weight_s;
    sc_signal<sc_int<ACC_WIDTH>> result_s[NUM_PE];
    sc_signal<bool> done_s[NUM_PE];

    SystolicArray dut("dut");
    dut.clk(clk_s);
    dut.reset(reset_s);
    dut.weight_in(weight_s);

    for (int i = 0; i < NUM_PE; i++) {
        dut.data_in[i](data_s[i]);
        dut.enable[i](enable_s[i]);
    }
    dut.connect(result_s, done_s);

    Testbench tb("tb");
    tb.clk(clk_s);
    tb.reset(reset_s);
    tb.weight(weight_s);

    for (int i = 0; i < NUM_PE; i++) {
        tb.data[i](data_s[i]);
        tb.result[i](result_s[i]);
        tb.done[i](done_s[i]);
        tb.enable[i](enable_s[i]);
    }

    sc_start();
    return 0;
}
