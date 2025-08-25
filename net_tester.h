#ifndef NET_TESTER_H
#define NET_TESTER_H


#include "neural_net.h"

class Net_Tester
{
private:
    Neural_Net* net;


    /*
    * Initializes the 2-D, (num_cases by num_inputs), array inputs
    * to be binary values equal to the case #
    * EX: With 4 inputs, inputs[11] would store binary of 11: {1,0,1,1}
    */
    void setup_inputs(double** inputs, int num_cases, int num_inputs);

    /**
     * Set up the expected outputs for AND, works for 2-B-1 network
     * Precondition: member var net is initialized; its inputs and outputs are being configured
     */
    void setup_AND();

    /**
     * Set up the expected outputs for OR, works for 2-B-1 network
     * Precondition: member var net is initialized; its inputs and outputs are being configured
     */
    void setup_OR();

    /**
     * Set up the expected outputs for XOR, works for 2-B-1 network
     * Precondition: member var net is initialized; its inputs and outputs are being configured
     */
    void setup_XOR();

    /**
     * Set up the expected outputs for AND, OR, and XOR, works for 2-B-3 network
     * Precondition: member var net is initialized; its inputs and outputs are being configured
     */
    void setup_AND_OR_XOR();

    /**
     * Set up more complex tester for a 3-B-4 network
     * Outputs are: XOR first two, AND of XOR function of first two with 3rd input,
     * OR of XOR function of first two with 3rd input, XOR of XOR function of first two with 3rd input
     * Precondition: member var net is initialized; its inputs and outputs are being configured
     */
    void setup_ABC_tester1();

    /**
     * Set up another complex tester, this time for a 4-B-3 network
     * Outputs are: XOR first two, OR of 3 and 4, AND of XOR and OR
     * Precondition: member var net is initialized; its inputs and outputs are being configured
     */
    void setup_ABC_tester2();


public:
    //Default constructor
    Net_Tester();

    //Parametrized constructor
    Net_Tester(Neural_Net*);

    //Destructor
    ~Net_Tester();


    void test_AND();

    void test_OR();

    void test_XOR();

    void test_AND_OR_XOR();

    void test_ABC_tester1();

    void test_ABC_tester2();
};

#endif