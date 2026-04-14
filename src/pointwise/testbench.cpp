#include "design.cpp"
#include <cstdio>

SC_MODULE(Testbench) {

    sc_out<bool> clk;
    sc_out<bool> reset;
    sc_out<bool> enable[NUM_PE];

    sc_out<sc_uint<DATA_WIDTH>> data[NUM_PE];
    sc_out<sc_uint<DATA_WIDTH>> weight;

    sc_in<sc_uint<ACC_WIDTH>> result[NUM_PE];
    sc_in<bool> done[NUM_PE];

    SC_CTOR(Testbench) {
        SC_THREAD(run);
    }

    void tick() {
        clk.write(1); wait(5, SC_NS);
        clk.write(0); wait(5, SC_NS);
    }

    void run_mode(bool gating,
                  int A[], int Wt[],
                  int out_results[NUM_PE],
                  int &active_pe) {

        for (int i = 0; i < NUM_PE; i++) {
            if (!gating)
                enable[i].write(true);
            else if (i == 0 || i == 4 || i == 8)
                enable[i].write(true);
            else
                enable[i].write(false);
        }

        if (!gating)
            active_pe = NUM_PE;
        else
            active_pe = 3;

        clk.write(0);
        reset.write(1);
        weight.write(0);

        for (int i = 0; i < NUM_PE; i++)
            data[i].write(0);

        tick(); tick();
        reset.write(0);
        tick();

        for (int k = 0; k < C; k++) {
            for (int p = 0; p < NUM_PE; p++)
                data[p].write(A[p*C + k]);

            weight.write(Wt[k]);
            tick();
        }

        for (int t = 0; t < 20 && !done[0].read(); t++)
            tick();

        tick();

        for (int i = 0; i < NUM_PE; i++)
            out_results[i] = result[i].read();
    }

    void print_matrix(const char* title, int arr[], bool show_mask, bool gating) {

        printf("\n--- %s (%d x %d) ---\n", title, H, W);

        for (int i = 0; i < H; i++) {
            printf("  [ ");
            for (int j = 0; j < W; j++) {

                int idx = i*W + j;

                if (show_mask && gating &&
                    !(idx == 0 || idx == 4 || idx == 8))
                    printf("  X  ");
                else
                    printf("%4d ", arr[idx]);
            }
            printf("]\n");
        }
    }

    void run() {

        int A[NUM_PE * C] = {
            2,0,1,   1,1,1,   3,2,1,
            0,1,2,   2,2,2,   1,0,1,
            3,1,0,   1,2,3,   2,2,2
        };

        int Wt[C] = {2,1,3};

        int results_full[NUM_PE];
        int results_gated[NUM_PE];

        int active_full = 0;
        int active_gated = 0;

        run_mode(false, A, Wt, results_full, active_full);
        print_matrix("FULL MODE OUTPUT", results_full, false, false);

        run_mode(true, A, Wt, results_gated, active_gated);

        printf("\n(Note: X = GATED PE)\n");
        print_matrix("GATED MODE OUTPUT", results_gated, true, true);

        bool pass = true;

        for (int i = 0; i < NUM_PE; i++) {

            int exp = 0;
            for (int k = 0; k < C; k++)
                exp += A[i*C + k] * Wt[k];

            if (results_full[i] != exp)
                pass = false;

            if ((i == 0 || i == 4 || i == 8)) {
                if (results_gated[i] != exp)
                    pass = false;
            }
        }

        printf("\n--- VALIDATION ---\n");
        printf("%s: Both modes correct!\n",
               pass ? "SUCCESS" : "FAIL");

        printf("\n=============================================\n");
        printf("        PPA COMPARISON REPORT\n");
        printf("=============================================\n");

        int latency_cycles = C + 1;
        int latency_ns = latency_cycles * 10;

        int physical_area = NUM_PE * 1500;

        float energy_full = active_full * C * 2.0;
        float energy_gated = active_gated * C * 2.0;

        float leakage_full = active_full * 5.0;
        float leakage_gated = active_gated * 5.0;

        float total_full = energy_full + leakage_full;
        float total_gated = energy_gated + leakage_gated;

        printf("[TIME ] Latency (both)        : %d ns (%d cycles)\n",
               latency_ns, latency_cycles);

        printf("[AREA ] Physical Area         : %d um^2 (constant)\n",
               physical_area);

        printf("[UTIL ] Active PEs (Full)     : %d / %d\n",
               active_full, NUM_PE);

        printf("[UTIL ] Active PEs (Gated)    : %d / %d\n",
               active_gated, NUM_PE);

        printf("[POWER] Full Energy           : %.2f pJ\n", total_full);
        printf("[POWER] Gated Energy          : %.2f pJ\n", total_gated);

        float util_reduction =
            (float)(active_full - active_gated) / active_full * 100.0;

        float power_saving =
            (float)(total_full - total_gated) / total_full * 100.0;

        printf("\n>>> UTILIZATION REDUCTION : %.2f%%\n", util_reduction);
        printf(">>> POWER SAVING          : %.2f%%\n", power_saving);

        printf("\n(Note: Area remains constant; gating reduces active switching units)\n");

        printf("=============================================\n\n");

        sc_stop();
    }
};

int sc_main(int argc, char* argv[]) {

    sc_signal<bool> clk_s, reset_s;
    sc_signal<bool> enable_s[NUM_PE];

    sc_signal<sc_uint<DATA_WIDTH>> data_s[NUM_PE];
    sc_signal<sc_uint<DATA_WIDTH>> weight_s;
    sc_signal<sc_uint<ACC_WIDTH>> result_s[NUM_PE];
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