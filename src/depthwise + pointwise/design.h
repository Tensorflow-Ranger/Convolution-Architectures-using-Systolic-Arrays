#ifndef DESIGN_H
#define DESIGN_H

#include <systemc.h>

#define H          3
#define W          3
#define K_SIZE     3
#define C_IN       3
#define C_OUT      4
#define DATA_WIDTH 8
#define ACC_WIDTH  32
#define NUM_PE     9 // 3x3 for spatial operations

SC_MODULE(PE) {
    sc_in<bool>                clk;
    sc_in<bool>                reset;
    sc_in<bool>                enable;
    sc_in<sc_int<DATA_WIDTH>>  data_in;
    sc_in<sc_int<DATA_WIDTH>>  weight_in;
    sc_out<sc_int<ACC_WIDTH>>  result_out;
    sc_out<bool>               done;

    SC_CTOR(PE) {
        SC_CTHREAD(proc, clk.pos());
        reset_signal_is(reset, true);
    }

    void proc() {
        result_out.write(0);
        done.write(false);
        wait();

        while (true) {
            if (!enable.read()) {
                done.write(false);
                wait();
                continue;
            }

            // Using long int instead of sc_uint to handle negative weights correctly!
            long int acc = 0;
            
            for (int k = 0; k < K_SIZE * K_SIZE; k++) {
                wait();
                if (enable.read()) {
                    acc += (int)data_in.read() *
                           (int)weight_in.read();
                }
            }

            result_out.write(acc);
            done.write(true);
            wait();
            done.write(false);
        }
    }
};

SC_MODULE(SystolicArray) {
    sc_in<bool>               clk;
    sc_in<bool>               reset;
    sc_in<bool>               enable[NUM_PE];
    sc_in<sc_int<DATA_WIDTH>> data_in[NUM_PE];
    sc_in<sc_int<DATA_WIDTH>> weight_in; // Shared weight broadcast for pointwise, individual for normal

    PE* pe[NUM_PE];

    SC_CTOR(SystolicArray) {
        for (int i = 0; i < NUM_PE; i++) {
            char nm[10];
            snprintf(nm, sizeof(nm), "PE_%d", i);

            pe[i] = new PE(nm);
            pe[i]->clk(clk);
            pe[i]->reset(reset);
            pe[i]->enable(enable[i]);
            pe[i]->data_in(data_in[i]);
            pe[i]->weight_in(weight_in);
        }
    }

    void connect(sc_signal<sc_int<ACC_WIDTH>> result_s[NUM_PE],
                 sc_signal<bool> done_s[NUM_PE]) {
        for (int i = 0; i < NUM_PE; i++) {
            pe[i]->result_out(result_s[i]);
            pe[i]->done(done_s[i]);
        }
    }

    ~SystolicArray() {
        for (int i = 0; i < NUM_PE; i++)
            delete pe[i];
    }
};

#endif // DESIGN_H
