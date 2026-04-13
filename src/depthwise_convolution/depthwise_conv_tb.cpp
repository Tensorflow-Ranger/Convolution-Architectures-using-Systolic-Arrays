#include <systemc.h>
#include <iostream>
#include "depthwise_conv.h"

// ============================================================================
// Testbench for DepthwiseConv
//
// Test 1 — 2-channel, 5×5 image, 3×3 kernel, dilation=1
//           (sanity check: depthwise at D=1 == two independent standard convs)
//
// Test 2 — 3-channel, 7×7 image, 3×3 kernel, dilation=2
//           (dilated depthwise, RGB-like input)
//
// Test 3 — 3-channel, 9×9 image, 3×3 kernel, dilation=3
//           (larger dilation, checks channel independence)
//
// Each test computes a per-channel golden reference in software and compares.
// ============================================================================

// --------------------------------------------------------------------------
// Software golden model for depthwise convolution
//
// For each channel ch independently:
//   out[ch][r][c] = Σ_{kr,kc}  kern[ch][kr][kc]
//                               * img[ch][r + kr*D][c + kc*D]
// --------------------------------------------------------------------------
template <int C, int IH, int IW, int K, int D>
void golden_depthwise_conv(
        const int img [C][IH][IW],
        const int kern[C][K][K],
        int       out [C][(IH - (K + (K-1)*(D-1)) + 1)]
                        [(IW - (K + (K-1)*(D-1)) + 1)])
{
    const int EK    = K + (K - 1) * (D - 1);
    const int OUT_H = IH - EK + 1;
    const int OUT_W = IW - EK + 1;

    for (int ch = 0; ch < C; ch++)
        for (int r = 0; r < OUT_H; r++)
            for (int c = 0; c < OUT_W; c++) {
                int sum = 0;
                for (int kr = 0; kr < K; kr++)
                    for (int kc = 0; kc < K; kc++)
                        sum += kern[ch][kr][kc]
                             * img[ch][r + kr * D][c + kc * D];
                out[ch][r][c] = sum;
            }
}

// --------------------------------------------------------------------------
// Compare DUT output vs golden, per channel
// --------------------------------------------------------------------------
template <int C, int H, int W>
bool check(const char* label,
           const int got     [C][H][W],
           const int expected[C][H][W])
{
    bool pass = true;
    for (int ch = 0; ch < C; ch++)
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                if (got[ch][r][c] != expected[ch][r][c]) {
                    std::cout << "  FAIL " << label
                              << " output[ch=" << ch << "][" << r << "][" << c << "] = "
                              << got[ch][r][c]
                              << "  expected " << expected[ch][r][c] << std::endl;
                    pass = false;
                }
    if (pass) std::cout << "  PASS " << label << std::endl;
    return pass;
}

// --------------------------------------------------------------------------
// Print a multi-channel 2-D array
// --------------------------------------------------------------------------
template <int C, int H, int W>
void print_tensor(const char* label, const int m[C][H][W]) {
    std::cout << label << ":" << std::endl;
    for (int ch = 0; ch < C; ch++) {
        std::cout << "  channel " << ch << ":" << std::endl;
        for (int r = 0; r < H; r++) {
            std::cout << "    [";
            for (int c = 0; c < W; c++) {
                if (c) std::cout << ", ";
                std::cout << m[ch][r][c];
            }
            std::cout << "]" << std::endl;
        }
    }
}

// ============================================================================
int sc_main(int /*argc*/, char* /*argv*/[]) {

    sc_clock      clk("clk", 10, SC_NS);
    sc_signal<bool> rst;

    // ------------------------------------------------------------------
    // Instantiate ALL DUTs before sc_start  (SystemC elaboration rule)
    // ------------------------------------------------------------------
    DepthwiseConv<2, 5, 5, 3, 1> dut1("DW_T1");
    dut1.clk(clk);  dut1.rst(rst);

    DepthwiseConv<3, 7, 7, 3, 2> dut2("DW_T2");
    dut2.clk(clk);  dut2.rst(rst);

    DepthwiseConv<3, 9, 9, 3, 3> dut3("DW_T3");
    dut3.clk(clk);  dut3.rst(rst);

    int tests_passed = 0, tests_total = 0;

    // ====================================================================
    // Test 1 — 2-channel, 5×5, K=3, D=1   →   output: 2×3×3
    // ====================================================================
    {
        std::cout << "\n=== Test 1: 2-channel 5x5 image, 3x3 kernel, dilation=1 ===" << std::endl;

        const int C=2, IH=5, IW=5, K=3, D=1;
        const int EK=K+(K-1)*(D-1), OH=IH-EK+1, OW=IW-EK+1;

        // Two distinct channels
        int image[C][IH][IW] = {
            // Channel 0 — ascending ramp
            {{ 1,  2,  3,  4,  5},
             { 6,  7,  8,  9, 10},
             {11, 12, 13, 14, 15},
             {16, 17, 18, 19, 20},
             {21, 22, 23, 24, 25}},
            // Channel 1 — descending ramp
            {{25, 24, 23, 22, 21},
             {20, 19, 18, 17, 16},
             {15, 14, 13, 12, 11},
             {10,  9,  8,  7,  6},
             { 5,  4,  3,  2,  1}}
        };

        // Two distinct kernels (Sobel-X for ch0, Sobel-Y for ch1)
        int kernel[C][K][K] = {
            {{ 1,  0, -1},
             { 1,  0, -1},
             { 1,  0, -1}},
            {{ 1,  2,  1},
             { 0,  0,  0},
             {-1, -2, -1}}
        };

        int golden[C][OH][OW];
        golden_depthwise_conv<C,IH,IW,K,D>(image, kernel, golden);
        print_tensor<C,OH,OW>("  Golden", golden);

        dut1.load_image(image);
        dut1.load_kernel(kernel);

        rst.write(true);
        sc_start(10, SC_NS);
        rst.write(false);

        dut1.run(rst);

        print_tensor<C,OH,OW>("  Output", dut1.output);
        tests_total++;
        if (check<C,OH,OW>("Test1", dut1.output, golden)) tests_passed++;
    }

    // ====================================================================
    // Test 2 — 3-channel, 7×7, K=3, D=2   →   output: 3×3×3
    // ====================================================================
    {
        std::cout << "\n=== Test 2: 3-channel 7x7 image, 3x3 kernel, dilation=2 ===" << std::endl;

        const int C=3, IH=7, IW=7, K=3, D=2;
        const int EK=K+(K-1)*(D-1), OH=IH-EK+1, OW=IW-EK+1;

        int image[C][IH][IW] = {
            // Channel 0
            {{ 1, 2, 3, 4, 5, 6, 7},
             { 8, 9,10,11,12,13,14},
             {15,16,17,18,19,20,21},
             {22,23,24,25,26,27,28},
             {29,30,31,32,33,34,35},
             {36,37,38,39,40,41,42},
             {43,44,45,46,47,48,49}},
            // Channel 1 — same ramp, offset by 1
            {{ 2, 3, 4, 5, 6, 7, 8},
             { 9,10,11,12,13,14,15},
             {16,17,18,19,20,21,22},
             {23,24,25,26,27,28,29},
             {30,31,32,33,34,35,36},
             {37,38,39,40,41,42,43},
             {44,45,46,47,48,49,50}},
            // Channel 2 — constant
            {{ 3, 3, 3, 3, 3, 3, 3},
             { 3, 3, 3, 3, 3, 3, 3},
             { 3, 3, 3, 3, 3, 3, 3},
             { 3, 3, 3, 3, 3, 3, 3},
             { 3, 3, 3, 3, 3, 3, 3},
             { 3, 3, 3, 3, 3, 3, 3},
             { 3, 3, 3, 3, 3, 3, 3}}
        };

        int kernel[C][K][K] = {
            // Channel 0 — Sobel-Y
            {{ 1,  2,  1},
             { 0,  0,  0},
             {-1, -2, -1}},
            // Channel 1 — identity-ish
            {{ 0,  0,  0},
             { 0,  1,  0},
             { 0,  0,  0}},
            // Channel 2 — sum (output should equal 9*3=27 everywhere)
            {{ 1,  1,  1},
             { 1,  1,  1},
             { 1,  1,  1}}
        };

        int golden[C][OH][OW];
        golden_depthwise_conv<C,IH,IW,K,D>(image, kernel, golden);
        print_tensor<C,OH,OW>("  Golden", golden);

        dut2.load_image(image);
        dut2.load_kernel(kernel);

        dut2.run(rst);

        print_tensor<C,OH,OW>("  Output", dut2.output);
        tests_total++;
        if (check<C,OH,OW>("Test2", dut2.output, golden)) tests_passed++;
    }

    // ====================================================================
    // Test 3 — 3-channel, 9×9, K=3, D=3   →   output: 3×3×3
    // ====================================================================
    {
        std::cout << "\n=== Test 3: 3-channel 9x9 image, 3x3 kernel, dilation=3 ===" << std::endl;

        const int C=3, IH=9, IW=9, K=3, D=3;
        const int EK=K+(K-1)*(D-1), OH=IH-EK+1, OW=IW-EK+1;

        int image[C][IH][IW] = {
            // Channel 0 — pseudo-random pattern from original test
            {{ 2, 0, 1, 3, 0, 2, 1, 0, 3},
             { 1, 3, 0, 2, 1, 0, 3, 2, 1},
             { 0, 2, 3, 1, 0, 3, 2, 1, 0},
             { 3, 1, 0, 2, 3, 1, 0, 2, 3},
             { 0, 2, 1, 3, 2, 0, 1, 3, 2},
             { 1, 0, 3, 1, 0, 2, 3, 1, 0},
             { 2, 3, 0, 2, 1, 3, 0, 2, 1},
             { 3, 1, 2, 0, 3, 1, 2, 0, 3},
             { 0, 2, 1, 3, 2, 0, 1, 3, 2}},
            // Channel 1 — rotated version
            {{ 3, 2, 1, 0, 3, 1, 2, 0, 2},
             { 1, 2, 3, 0, 1, 2, 0, 3, 1},
             { 0, 1, 2, 3, 2, 1, 3, 0, 2},
             { 3, 2, 1, 0, 3, 2, 1, 0, 3},
             { 2, 3, 1, 0, 2, 3, 1, 0, 2},
             { 0, 1, 2, 3, 0, 1, 2, 3, 0},
             { 1, 2, 3, 0, 1, 2, 3, 0, 1},
             { 3, 0, 1, 2, 3, 0, 1, 2, 3},
             { 2, 3, 0, 1, 2, 3, 0, 1, 2}},
            // Channel 2 — all ones (easy to verify manually)
            {{ 1, 1, 1, 1, 1, 1, 1, 1, 1},
             { 1, 1, 1, 1, 1, 1, 1, 1, 1},
             { 1, 1, 1, 1, 1, 1, 1, 1, 1},
             { 1, 1, 1, 1, 1, 1, 1, 1, 1},
             { 1, 1, 1, 1, 1, 1, 1, 1, 1},
             { 1, 1, 1, 1, 1, 1, 1, 1, 1},
             { 1, 1, 1, 1, 1, 1, 1, 1, 1},
             { 1, 1, 1, 1, 1, 1, 1, 1, 1},
             { 1, 1, 1, 1, 1, 1, 1, 1, 1}}
        };

        int kernel[C][K][K] = {
            // Channel 0 — from original test 3
            {{ 1, -1,  1},
             {-1,  2, -1},
             { 1, -1,  1}},
            // Channel 1 — box filter
            {{ 1,  1,  1},
             { 1,  1,  1},
             { 1,  1,  1}},
            // Channel 2 — sum kernel; output should = 9 everywhere
            {{ 1,  1,  1},
             { 1,  1,  1},
             { 1,  1,  1}}
        };

        int golden[C][OH][OW];
        golden_depthwise_conv<C,IH,IW,K,D>(image, kernel, golden);
        print_tensor<C,OH,OW>("  Golden", golden);

        dut3.load_image(image);
        dut3.load_kernel(kernel);

        dut3.run(rst);

        print_tensor<C,OH,OW>("  Output", dut3.output);
        tests_total++;
        if (check<C,OH,OW>("Test3", dut3.output, golden)) tests_passed++;
    }

    // ====================================================================
    // Summary
    // ====================================================================
    std::cout << "\n=== SUMMARY: " << tests_passed << "/" << tests_total
              << " tests passed ===" << std::endl;

    return (tests_passed == tests_total) ? 0 : 1;
}
