#include "net_tester.h"

void Net_Tester::setup_AND()
{
    double** inputs = new double*[4];
    double** outputs = new double*[4];
    for(int i = 0; i < 4; i++)
    {
        inputs[i] = new double[2];
        outputs[i] = new double[1];
    }
    //Case 1
    inputs[0][0] = 0.0;
    inputs[0][1] = 0.0;
    outputs[0][0] = 0.0;

    //Case2
    inputs[1][0] = 0.0;
    inputs[1][1] = 1.0;
    outputs[1][0] = 0.0;

    //Case3
    inputs[2][0] = 1.0;
    inputs[2][1] = 0.0;
    outputs[2][0] = 0.0;

    //Case4
    inputs[3][0] = 1.0;
    inputs[3][1] = 1.0;
    outputs[3][0] = 1.0;
    this->net->set_possible_inputs(inputs, 4);
    this->net->set_expected_outputs(outputs, 4);
}


void Net_Tester::setup_OR()
{
    double** inputs = new double*[4];
    double** outputs = new double*[4];
    for(int i = 0; i < 4; i++)
    {
        inputs[i] = new double[2];
        outputs[i] = new double[1];
    }
    //Case 1
    inputs[0][0] = 0.0;
    inputs[0][1] = 0.0;
    outputs[0][0] = 0.0;

    //Case2
    inputs[1][0] = 0.0;
    inputs[1][1] = 1.0;
    outputs[1][0] = 1.0;

    //Case3
    inputs[2][0] = 1.0;
    inputs[2][1] = 0.0;
    outputs[2][0] = 1.0;

    //Case4
    inputs[3][0] = 1.0;
    inputs[3][1] = 1.0;
    outputs[3][0] = 1.0;
    this->net->set_possible_inputs(inputs, 4);
    this->net->set_expected_outputs(outputs, 4);
}


void Net_Tester::setup_XOR()
{

}


void Net_Tester::setup_AND_OR_XOR()
{

}


void Net_Tester::setup_ABC_tester1()
{

}


void Net_Tester::setup_ABC_tester2()
{

}



Net_Tester::Net_Tester()
{
    this->net = nullptr;
}


Net_Tester::Net_Tester(Neural_Net* n)
{
    this->net = n;
}


Net_Tester::~Net_Tester()
{
    if(this->net != nullptr)
    {
        delete [] this->net;
    }
}


void Net_Tester::test_AND()
{
    Net_Tester::setup_AND();
    cout << endl << "—————————————————————" << endl << "Testing AND" << endl;
    cout << "—————————————————————" << endl;
    this->net->train_network();
    cout << "—————————————————————" << endl << endl << endl;
    this->net->run_network(true, false);
    cout << endl << "—————————————————————" << endl << "End Testing AND" << endl;
    cout << "—————————————————————" << endl;
}


void Net_Tester::test_OR()
{
    Net_Tester::setup_AND();
    cout << endl << "—————————————————————" << endl << "Testing OR" << endl;
    cout << "—————————————————————" << endl;
    this->net->train_network();
    cout << "—————————————————————" << endl << endl << endl;
    this->net->run_network(true, false);
    cout << endl << "—————————————————————" << endl << "End Testing OR" << endl;
    cout << "—————————————————————" << endl;
}