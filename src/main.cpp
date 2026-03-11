#include <systemc.h>
#include "pe.h"

int sc_main(int argc, char* argv[]) {
    // 1. Create Signals (Wires)
    sc_clock clk("clk", 10, SC_NS); // 10ns clock period
    sc_signal<bool> rst;
    
    sc_signal<int> sig_in_act;
    sc_signal<int> sig_in_psum;
    sc_signal<int> sig_out_act;
    sc_signal<int> sig_out_psum;

    // 2. Instantiate the PE
    PE pe0("PE_0_0");
    pe0.clk(clk);
    pe0.rst(rst);
    pe0.in_act(sig_in_act);
    pe0.in_psum(sig_in_psum);
    pe0.out_act(sig_out_act);
    pe0.out_psum(sig_out_psum);

    // 3. Setup Waveform Tracing
    sc_trace_file *wf = sc_create_vcd_trace_file("waveforms");
    sc_trace(wf, clk, "clk");
    sc_trace(wf, rst, "rst");
    sc_trace(wf, sig_in_act, "in_act");
    sc_trace(wf, sig_out_psum, "out_psum");

    // 4. Run the Simulation
    cout << "Starting Simulation..." << endl;
    
    // Reset phase
    rst.write(true);
    sig_in_act.write(0);
    sig_in_psum.write(0);
    sc_start(20, SC_NS); 

    // Active phase
    rst.write(false);
    
    // Cycle 1: Feed activation '5'
    sig_in_act.write(5);
    sig_in_psum.write(10); // initial partial sum
    sc_start(10, SC_NS);
    cout << "Cycle 1 -> Out Psum: " << sig_out_psum.read() << endl;

    // Cycle 2: Feed activation '3'
    sig_in_act.write(3);
    sig_in_psum.write(0);
    sc_start(10, SC_NS);
    cout << "Cycle 2 -> Out Psum: " << sig_out_psum.read() << endl;

    // 5. Cleanup
    sc_close_vcd_trace_file(wf);
    cout << "Simulation Complete. Check waveforms.vcd!" << endl;

    return 0;
}