#include <systemc.h>
#include <iostream>
#include "depthwise_conv.h"

// ==========================================================
// PRINT TENSOR
// ==========================================================
template <int C, int H, int W>
void print_tensor(const char* label, const int m[C][H][W]) {
    std::cout << label << std::endl;
    for (int ch = 0; ch < C; ch++) {
        std::cout << "Channel " << ch << std::endl;
        for (int r = 0; r < H; r++) {
            for (int c = 0; c < W; c++)
                std::cout << m[ch][r][c] << " ";
            std::cout << std::endl;
        }
    }
}

// ==========================================================
// UTILIZATION
// ==========================================================
template <typename DUT>
void print_utilization(DUT& dut, int K, const std::string& label) {
    std::cout << "\n--- OPTIMIZATION REPORT (" << label << ") ---" << std::endl;

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            float util = 0;
            if (dut.sa->pe[i][j]->total_cycles != 0)
                util = (float)dut.sa->pe[i][j]->active_cycles /
                       dut.sa->pe[i][j]->total_cycles * 100;

            std::cout << "PE(" << i << "," << j << ") Utilization: "
                      << util << "%" << std::endl;
        }
    }
}

// ==========================================================
// PPA
// ==========================================================
template <typename DUT>
void print_ppa(DUT& dut, int K, const std::string& label) {

    std::cout << "\n===============================================" << std::endl;
    std::cout << "     HARDWARE PPA ANALYSIS REPORT (" << label << ")" << std::endl;
    std::cout << "===============================================" << std::endl;

    const float CLOCK_PERIOD_NS = 10.0;
    const float MAC_ENERGY_PJ = 5.0;
    const float IDLE_ENERGY_PJ = 1.0;
    const float PE_AREA_UM2 = 1500.0;

    int global_active_cycles = 0;
    int global_total_cycles = 0;

    for (int r = 0; r < K; r++) {
        for (int c = 0; c < K; c++) {
            global_active_cycles += dut.sa->pe[r][c]->active_cycles;

            if (dut.sa->pe[r][c]->total_cycles > global_total_cycles)
                global_total_cycles = dut.sa->pe[r][c]->total_cycles;
        }
    }

    float total_time_ns = global_total_cycles * CLOCK_PERIOD_NS;
    int total_pes = K * K;
    float total_area = total_pes * PE_AREA_UM2;

    float dynamic_energy = global_active_cycles * MAC_ENERGY_PJ;
    float static_energy = (total_pes * global_total_cycles) * IDLE_ENERGY_PJ;
    float total_energy = dynamic_energy + static_energy;

    float avg_power = total_energy / total_time_ns;

    std::cout << "[TIME] Total Latency:      "
              << total_time_ns << " ns (" << global_total_cycles << " cycles)" << std::endl;

    std::cout << "[AREA] Total Array Area:   "
              << total_area << " um^2 (" << total_pes << " PEs)" << std::endl;

    std::cout << "[POWER] Dynamic Energy:    " << dynamic_energy << " pJ" << std::endl;
    std::cout << "[POWER] Static Leakage:    " << static_energy << " pJ" << std::endl;
    std::cout << "[POWER] Total Energy:      " << total_energy << " pJ" << std::endl;
    std::cout << "[POWER] Avg Power Consump: " << avg_power << " mW" << std::endl;

    std::cout << "===============================================\n" << std::endl;
}

// ==========================================================
// GOLDEN MODEL
// ==========================================================
template <int C, int IH, int IW, int K, int D>
void golden_depthwise_conv(
        const int img[C][IH][IW],
        const int kern[C][K][K],
        int out[C][(IH - (K + (K-1)*(D-1)) + 1)]
                  [(IW - (K + (K-1)*(D-1)) + 1)])
{
    const int EK = K + (K - 1) * (D - 1);
    const int OH = IH - EK + 1;
    const int OW = IW - EK + 1;

    for (int ch = 0; ch < C; ch++)
        for (int r = 0; r < OH; r++)
            for (int c = 0; c < OW; c++) {
                int sum = 0;
                for (int kr = 0; kr < K; kr++)
                    for (int kc = 0; kc < K; kc++)
                        sum += kern[ch][kr][kc]
                             * img[ch][r + kr * D][c + kc * D];
                out[ch][r][c] = sum;
            }
}

// ==========================================================
// CHECK
// ==========================================================
template <int C, int H, int W>
bool check(const char* label,
           const int got[C][H][W],
           const int expected[C][H][W])
{
    bool pass = true;
    for (int ch = 0; ch < C; ch++)
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                if (got[ch][r][c] != expected[ch][r][c])
                    pass = false;

    if (pass) std::cout << "PASS " << label << std::endl;
    else std::cout << "FAIL " << label << std::endl;

    return pass;
}

// ==========================================================
// MAIN
// ==========================================================
int sc_main(int, char**) {

    sc_clock clk("clk", 10, SC_NS);
    sc_signal<bool> rst;

    // ALL DUTs at top (IMPORTANT)
    DepthwiseConv<2,5,5,3,1> dut1("DW1");
    DepthwiseConv<3,7,7,3,2> dut2("DW2");

    dut1.clk(clk); dut1.rst(rst);
    dut2.clk(clk); dut2.rst(rst);

    // ================= TEST 1 =================
    {
        const int C=2, IH=5, IW=5, K=3, D=1;
        const int EK=K+(K-1)*(D-1), OH=IH-EK+1, OW=IW-EK+1;

        int image[C][IH][IW] = {
            {{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15},{16,17,18,19,20},{21,22,23,24,25}},
            {{25,24,23,22,21},{20,19,18,17,16},{15,14,13,12,11},{10,9,8,7,6},{5,4,3,2,1}}
        };

        int kernel[C][K][K] = {
            {{1,0,-1},{1,0,-1},{1,0,-1}},
            {{1,2,1},{0,0,0},{-1,-2,-1}}
        };

        int golden[C][OH][OW];
        golden_depthwise_conv<C,IH,IW,K,D>(image, kernel, golden);

        dut1.load_image(image);
        dut1.load_kernel(kernel);

        rst.write(true); sc_start(10, SC_NS);
        rst.write(false);

        dut1.run(rst);

        print_tensor<C,OH,OW>("Golden:", golden);
        print_tensor<C,OH,OW>("Output:", dut1.output);

        check<C,OH,OW>("Test1", dut1.output, golden);
        print_utilization(dut1, K, "Test1");
        print_ppa(dut1, K, "Test1");
    }

    // ================= TEST 2 =================
    {
        const int C=3, IH=7, IW=7, K=3, D=2;

        int image[C][IH][IW];

        // better input (NOT zeros)
        for(int ch=0; ch<C; ch++)
        for(int i=0;i<IH;i++)
        for(int j=0;j<IW;j++)
            image[ch][i][j] = (i+j+ch)%5 + 1;

        int kernel[C][K][K] = {
            {{1,1,1},{1,1,1},{1,1,1}},
            {{0,1,0},{1,2,1},{0,1,0}},
            {{1,0,-1},{1,0,-1},{1,0,-1}}
        };

        dut2.load_image(image);
        dut2.load_kernel(kernel);

        rst.write(true); sc_start(10, SC_NS);
        rst.write(false);

        dut2.run(rst);

        print_utilization(dut2, K, "Test2");
        print_ppa(dut2, K, "Test2");
    }

    return 0;
}
