#include <systemc.h>
#include <iostream>
#include <cstdlib>
#include "dilated_conv.h"

// ============================================================================
// Testbench for dilated convolution
//
// Test 1 — 5x5 image, 3x3 kernel, dilation = 1 (standard conv, sanity check)
// Test 2 — 7x7 image, 3x3 kernel, dilation = 2
// Test 3 — 9x9 image, 3x3 kernel, dilation = 3
//
// Each test computes a golden reference in software and compares.
// ============================================================================

// ---------- Software golden model ----------
template <int IH, int IW, int K, int D>
void golden_dilated_conv(const int img[IH][IW],
                         const int kern[K][K],
                         int out[(IH - (K + (K-1)*(D-1)) + 1)][(IW - (K + (K-1)*(D-1)) + 1)])
{
    const int EK    = K + (K - 1) * (D - 1);
    const int OUT_H = IH - EK + 1;
    const int OUT_W = IW - EK + 1;

    for (int r = 0; r < OUT_H; r++) {
        for (int c = 0; c < OUT_W; c++) {
            int sum = 0;
            for (int kr = 0; kr < K; kr++)
                for (int kc = 0; kc < K; kc++)
                    sum += kern[kr][kc] * img[r + kr * D][c + kc * D];
            out[r][c] = sum;
        }
    }
}

// ---------- Helper: compare two 2-D arrays ----------
template <int H, int W>
bool check(const char* label, const int got[H][W], const int expected[H][W]) {
    bool pass = true;
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            if (got[r][c] != expected[r][c]) {
                std::cout << "  FAIL " << label
                          << " output[" << r << "][" << c << "] = " << got[r][c]
                          << "  expected " << expected[r][c] << std::endl;
                pass = false;
            }
        }
    }
    if (pass)
        std::cout << "  PASS " << label << std::endl;
    return pass;
}

// ---------- Helper: print 2-D array ----------
template <int H, int W>
void print_matrix(const char* label, const int m[H][W]) {
    std::cout << label << ":" << std::endl;
    for (int r = 0; r < H; r++) {
        std::cout << "  [";
        for (int c = 0; c < W; c++) {
            if (c) std::cout << ", ";
            std::cout << m[r][c];
        }
        std::cout << "]" << std::endl;
    }
}

// ============================================================================
int sc_main(int argc, char* argv[]) {

    sc_clock clk("clk", 10, SC_NS);
    sc_signal<bool> rst;

    // ------------------------------------------------------------------
    // Instantiate ALL DUTs before any sc_start (SystemC elaboration rule)
    // ------------------------------------------------------------------
    DilatedConv<5, 5, 3, 1> dut1("DilatedConv_T1");
    dut1.clk(clk);
    dut1.rst(rst);

    DilatedConv<7, 7, 3, 2> dut2("DilatedConv_T2");
    dut2.clk(clk);
    dut2.rst(rst);

    DilatedConv<9, 9, 3, 3> dut3("DilatedConv_T3");
    dut3.clk(clk);
    dut3.rst(rst);

    int tests_passed = 0;
    int tests_total  = 0;

    // ====================================================================
    // Test 1: 5x5 image, 3x3 kernel, dilation = 1 (standard convolution)
    // Output: 3x3
    // ====================================================================
    {
        std::cout << "\n=== Test 1: 5x5 image, 3x3 kernel, dilation=1 ===" << std::endl;

        const int IH = 5, IW = 5, K = 3, D = 1;
        const int EK = K + (K-1)*(D-1);
        const int OH = IH - EK + 1, OW = IW - EK + 1;

        int image[IH][IW] = {
            { 1,  2,  3,  4,  5},
            { 6,  7,  8,  9, 10},
            {11, 12, 13, 14, 15},
            {16, 17, 18, 19, 20},
            {21, 22, 23, 24, 25}
        };
        int kernel[K][K] = {
            {1, 0, -1},
            {1, 0, -1},
            {1, 0, -1}
        };

        int golden[OH][OW];
        golden_dilated_conv<IH, IW, K, D>(image, kernel, golden);
        print_matrix<OH, OW>("  Golden", golden);

        dut1.load_image(image);
        dut1.load_kernel(kernel);

        rst.write(true);
        sc_start(10, SC_NS);
        rst.write(false);

        dut1.run(rst);

        print_matrix<OH, OW>("  Output", dut1.output);
        tests_total++;
        if (check<OH, OW>("Test1", dut1.output, golden)) tests_passed++;
    }

    // ====================================================================
    // Test 2: 7x7 image, 3x3 kernel, dilation = 2
    // Effective kernel: 5x5.   Output: 3x3
    // ====================================================================
    {
        std::cout << "\n=== Test 2: 7x7 image, 3x3 kernel, dilation=2 ===" << std::endl;

        const int IH = 7, IW = 7, K = 3, D = 2;
        const int EK = K + (K-1)*(D-1);
        const int OH = IH - EK + 1, OW = IW - EK + 1;

        int image[IH][IW] = {
            { 1,  2,  3,  4,  5,  6,  7},
            { 8,  9, 10, 11, 12, 13, 14},
            {15, 16, 17, 18, 19, 20, 21},
            {22, 23, 24, 25, 26, 27, 28},
            {29, 30, 31, 32, 33, 34, 35},
            {36, 37, 38, 39, 40, 41, 42},
            {43, 44, 45, 46, 47, 48, 49}
        };
        int kernel[K][K] = {
            {1, 2, 1},
            {0, 0, 0},
            {-1, -2, -1}
        };

        int golden[OH][OW];
        golden_dilated_conv<IH, IW, K, D>(image, kernel, golden);
        print_matrix<OH, OW>("  Golden", golden);

        dut2.load_image(image);
        dut2.load_kernel(kernel);

        dut2.run(rst);

        print_matrix<OH, OW>("  Output", dut2.output);
        tests_total++;
        if (check<OH, OW>("Test2", dut2.output, golden)) tests_passed++;
    }

    // ====================================================================
    // Test 3: 9x9 image, 3x3 kernel, dilation = 3
    // Effective kernel: 7x7.   Output: 3x3
    // ====================================================================
    {
        std::cout << "\n=== Test 3: 9x9 image, 3x3 kernel, dilation=3 ===" << std::endl;

        const int IH = 9, IW = 9, K = 3, D = 3;
        const int EK = K + (K-1)*(D-1);
        const int OH = IH - EK + 1, OW = IW - EK + 1;

        int image[IH][IW] = {
            { 2,  0,  1,  3,  0,  2,  1,  0,  3},
            { 1,  3,  0,  2,  1,  0,  3,  2,  1},
            { 0,  2,  3,  1,  0,  3,  2,  1,  0},
            { 3,  1,  0,  2,  3,  1,  0,  2,  3},
            { 0,  2,  1,  3,  2,  0,  1,  3,  2},
            { 1,  0,  3,  1,  0,  2,  3,  1,  0},
            { 2,  3,  0,  2,  1,  3,  0,  2,  1},
            { 3,  1,  2,  0,  3,  1,  2,  0,  3},
            { 0,  2,  1,  3,  2,  0,  1,  3,  2}
        };
        int kernel[K][K] = {
            { 1, -1,  1},
            {-1,  2, -1},
            { 1, -1,  1}
        };

        int golden[OH][OW];
        golden_dilated_conv<IH, IW, K, D>(image, kernel, golden);
        print_matrix<OH, OW>("  Golden", golden);

        dut3.load_image(image);
        dut3.load_kernel(kernel);

        dut3.run(rst);

        print_matrix<OH, OW>("  Output", dut3.output);
        tests_total++;
        if (check<OH, OW>("Test3", dut3.output, golden)) tests_passed++;
    }

    // ====================================================================
    // Summary
    // ====================================================================
    std::cout << "\n=== SUMMARY: " << tests_passed << "/" << tests_total
              << " tests passed ===" << std::endl;

    return (tests_passed == tests_total) ? 0 : 1;
}
