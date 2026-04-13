#include <systemc.h>
#include <iostream>
#include "standard_conv.h"

// ============================================================================
// Testbench for StandardConv
//
// Test 1 — Single-channel, 5×5 image, 3×3 kernel
//           (basic sanity check)
//
// Test 2 — Single-channel, 7×7 image, 3×3 kernel
//           (larger input, tests boundary handling)
//
// Test 3 — Single-channel, 9×9 image, 5×5 kernel
//           (larger kernel, tests multiple accumulations)
//
// Each test computes a golden reference in software and compares.
// ============================================================================

// --------------------------------------------------------------------------
// Software golden model for standard convolution
//
// Simple 2D convolution:
//   out[r][c] = Σ_{kr,kc}  kern[kr][kc] * img[r + kr][c + kc]
// --------------------------------------------------------------------------
template <int IH, int IW, int K>
void golden_standard_conv(
        const int img [IH][IW],
        const int kern[K][K],
        int       out [(IH - K + 1)][(IW - K + 1)])
{
    const int OUT_H = IH - K + 1;
    const int OUT_W = IW - K + 1;

    for (int r = 0; r < OUT_H; r++)
        for (int c = 0; c < OUT_W; c++) {
            int sum = 0;
            for (int kr = 0; kr < K; kr++)
                for (int kc = 0; kc < K; kc++)
                    sum += kern[kr][kc] * img[r + kr][c + kc];
            out[r][c] = sum;
        }
}

// --------------------------------------------------------------------------
// Compare DUT output vs golden
// --------------------------------------------------------------------------
template <int H, int W>
bool check(const char* label,
           const int got     [H][W],
           const int expected[H][W])
{
    bool pass = true;
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            if (got[r][c] != expected[r][c]) {
                std::cout << "  FAIL " << label
                          << " output[" << r << "][" << c << "] = "
                          << got[r][c]
                          << "  expected " << expected[r][c] << std::endl;
                pass = false;
            }
    if (pass) std::cout << "  PASS " << label << std::endl;
    return pass;
}

// --------------------------------------------------------------------------
// Print a 2-D array
// --------------------------------------------------------------------------
template <int H, int W>
void print_tensor(const char* label, const int m[H][W]) {
    std::cout << label << ":" << std::endl;
    for (int r = 0; r < H; r++) {
        std::cout << "    [";
        for (int c = 0; c < W; c++) {
            if (c) std::cout << ", ";
            std::cout << m[r][c];
        }
        std::cout << "]" << std::endl;
    }
}

// ============================================================================
int sc_main(int /*argc*/, char* /*argv*/[]) {

    sc_clock      clk("clk", 10, SC_NS);
    sc_signal<bool> rst;

    // ------------------------------------------------------------------
    // Instantiate ALL DUTs before sc_start  (SystemC elaboration rule)
    // ------------------------------------------------------------------
    StandardConv<5, 5, 3> dut1("StandardConv_T1");
    dut1.clk(clk);  dut1.rst(rst);

    StandardConv<7, 7, 3> dut2("StandardConv_T2");
    dut2.clk(clk);  dut2.rst(rst);

    StandardConv<9, 9, 5> dut3("StandardConv_T3");
    dut3.clk(clk);  dut3.rst(rst);

    int tests_passed = 0, tests_total = 0;

    // ====================================================================
    // Test 1 — 5×5 image, 3×3 kernel  →  output: 3×3
    // ====================================================================
    {
        std::cout << "\n=== Test 1: 5x5 image, 3x3 kernel ===" << std::endl;

        const int IH=5, IW=5, K=3;
        const int OH=IH-K+1, OW=IW-K+1;

        // Ascending ramp image
        int image[IH][IW] = {
            { 1,  2,  3,  4,  5},
            { 6,  7,  8,  9, 10},
            {11, 12, 13, 14, 15},
            {16, 17, 18, 19, 20},
            {21, 22, 23, 24, 25}
        };

        // Sobel-X kernel
        int kernel[K][K] = {
            { 1,  0, -1},
            { 1,  0, -1},
            { 1,  0, -1}
        };

        int golden[OH][OW];
        golden_standard_conv<IH,IW,K>(image, kernel, golden);
        print_tensor<OH,OW>("  Golden", golden);

        // Load and run DUT
        dut1.load_image(image);
        dut1.load_kernel(kernel);
        dut1.run(rst);

        // Compare
        tests_total++;
        if (check<OH,OW>("Test 1", dut1.output, golden))
            tests_passed++;
    }

    // ====================================================================
    // Test 2 — 7×7 image, 3×3 kernel  →  output: 5×5
    // ====================================================================
    {
        std::cout << "\n=== Test 2: 7x7 image, 3x3 kernel ===" << std::endl;

        const int IH=7, IW=7, K=3;
        const int OH=IH-K+1, OW=IW-K+1;

        // Pattern image
        int image[IH][IW] = {
            { 1,  2,  3,  4,  5,  6,  7},
            { 8,  9, 10, 11, 12, 13, 14},
            {15, 16, 17, 18, 19, 20, 21},
            {22, 23, 24, 25, 26, 27, 28},
            {29, 30, 31, 32, 33, 34, 35},
            {36, 37, 38, 39, 40, 41, 42},
            {43, 44, 45, 46, 47, 48, 49}
        };

        // Sobel-Y kernel
        int kernel[K][K] = {
            { 1,  2,  1},
            { 0,  0,  0},
            {-1, -2, -1}
        };

        int golden[OH][OW];
        golden_standard_conv<IH,IW,K>(image, kernel, golden);
        print_tensor<OH,OW>("  Golden", golden);

        // Load and run DUT
        dut2.load_image(image);
        dut2.load_kernel(kernel);
        dut2.run(rst);

        // Compare
        tests_total++;
        if (check<OH,OW>("Test 2", dut2.output, golden))
            tests_passed++;
    }

    // ====================================================================
    // Test 3 — 9×9 image, 5×5 kernel  →  output: 5×5
    // ====================================================================
    {
        std::cout << "\n=== Test 3: 9x9 image, 5x5 kernel ===" << std::endl;

        const int IH=9, IW=9, K=5;
        const int OH=IH-K+1, OW=IW-K+1;

        // Uniform image
        int image[IH][IW] = {};
        for (int i = 0; i < IH; i++)
            for (int j = 0; j < IW; j++)
                image[i][j] = i + j; // Diagonal pattern

        // All-ones kernel
        int kernel[K][K] = {};
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++)
                kernel[i][j] = 1;

        int golden[OH][OW];
        golden_standard_conv<IH,IW,K>(image, kernel, golden);
        print_tensor<OH,OW>("  Golden", golden);

        // Load and run DUT
        dut3.load_image(image);
        dut3.load_kernel(kernel);
        dut3.run(rst);

        // Compare
        tests_total++;
        if (check<OH,OW>("Test 3", dut3.output, golden))
            tests_passed++;
    }

    // ====================================================================
    // Summary
    // ====================================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "Tests passed: " << tests_passed << " / " << tests_total << std::endl;
    std::cout << "========================================" << std::endl;

    return (tests_passed == tests_total) ? 0 : 1;
}
