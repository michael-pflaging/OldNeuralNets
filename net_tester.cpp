#include "net_tester.h"

/*
* Sets up the inputs array for testing
* The inputs array is a 2-D array, where the dimensions are the number of test cases by
* the number of input neurons in the network.
* Each test case has a unique combination of 0s and 1s which are used as boolean values to generate
* the expected output
* Therefore, the 0s and 1s of each test case is actually the binary representation of which
* number test case it is
* 
* (yes I was proud of figuring this one out and coming up with a sleek solution for it, and yes
* I did use to hardcode literally every test case in the OG code, the README has a warning section
* about the old code for a reason)
*/
void Net_Tester::setup_inputs(double** inputs, int num_cases, int num_inputs)
{
    for(int i = 0; i < num_cases; i++)
    {
        int ctr = i;
        //This initiliazes the 4 expected inputs to the binary value of i
        for(int j = num_inputs - 1; j >= 0; j--)
        {
            inputs[i][j] = ctr % 2;
            ctr /= 2;
        }
    }
}


/*
* One of the OG test cases, designed to test a 2-B-1 network
* Tests the logical AND combination of the two input activations,
* and measured in the singular output activation
* 
* This is a linearly separable problem, if this fails something is really wrong
* 
* The number of hidden layers is not required to be 1
*/
void Net_Tester::setup_AND()
{
    //Allocate memory for expected inputs/outputs. (2-B-1) network
    int num_cases = 4;
    double** inputs = new double*[num_cases];
    double** outputs = new double*[num_cases];
    for(int i = 0; i < 4; i++)
    {
        inputs[i] = new double[2];
        outputs[i] = new double[1];
    }


    //Initialize values in expected input/outputs
    Net_Tester::setup_inputs(inputs, num_cases, 2);
    //1st & only output stores AND of inputs
    for(int i = 0; i < num_cases; i++)
        outputs[i][0] = inputs[i][0] && inputs[i][1];


    //Pass expected inputs/outputs to be deep copied
    this->net->set_expected_inputs(inputs, num_cases);
    this->net->set_expected_outputs(outputs, num_cases);


    //Free memory allocated to expected inputs/outputs
    for(int i = 0; i < num_cases; i++)
    {
        delete [] inputs[i];
        delete [] outputs[i];
    }
    delete [] inputs;
    delete [] outputs;
}


/*
* One of the OG test cases, designed to test a 2-B-1 network
* Tests the logical OR combination of the two input activations,
* and measured in the singular output activation
* 
* This is a linearly separable problem, if this fails something is really wrong
* 
* The number of hidden layers is not required to be 1
*/
void Net_Tester::setup_OR()
{
    //Allocate memory for expected inputs/outputs. (2-B-1) network
    int num_cases = 4;
    double** inputs = new double*[num_cases];
    double** outputs = new double*[num_cases];
    for(int i = 0; i < 4; i++)
    {
        inputs[i] = new double[2];
        outputs[i] = new double[1];
    }


    //Initialize values in expected input/outputs
    Net_Tester::setup_inputs(inputs, num_cases, 2);
    //1st & only output stores OR of inputs
    for(int i = 0; i < num_cases; i++)
        outputs[i][0] = inputs[i][0] || inputs[i][1];


    //Pass expected inputs/outputs to be deep copied
    this->net->set_expected_inputs(inputs, num_cases);
    this->net->set_expected_outputs(outputs, num_cases);


    //Free memory allocated to expected inputs/outputs
    for(int i = 0; i < num_cases; i++)
    {
        delete [] inputs[i];
        delete [] outputs[i];
    }
    delete [] inputs;
    delete [] outputs;
}


/*
* One of the OG test cases, designed to test a 2-B-1 network
* Tests the logical XOR combination of the two input activations,
* and measured in the singular output activation
* 
* This is *not* a linearly separable problem, this was the original source
* of pain in suffering back when I learned this stuff in HS.
* 
* The number of hidden layers is not required to be 1
*/
void Net_Tester::setup_XOR()
{
    //Allocate memory for expected inputs/outputs. (2-B-1) network
    int num_cases = 4;
    double** inputs = new double*[num_cases];
    double** outputs = new double*[num_cases];
    for(int i = 0; i < 4; i++)
    {
        inputs[i] = new double[2];
        outputs[i] = new double[1];
    }


    //Initialize values in expected input/outputs
    Net_Tester::setup_inputs(inputs, num_cases, 2);
    //1st & only output stores XOR of inputs
    for(int i = 0; i < num_cases; i++)
        outputs[i][0] = (inputs[i][0] || inputs[i][1]) && (inputs[i][0] != inputs[i][1]);


    //Pass expected inputs/outputs to be deep copied
    this->net->set_expected_inputs(inputs, num_cases);
    this->net->set_expected_outputs(outputs, num_cases);


    //Free memory allocated to expected inputs/outputs
    for(int i = 0; i < num_cases; i++)
    {
        delete [] inputs[i];
        delete [] outputs[i];
    }
    delete [] inputs;
    delete [] outputs;
}


/*
* One of the OG test cases, designed to test a 2-B-3 network
* The first output is expected to be the AND of inputs 1 & 2
* The second output is expected to be the OR of inputs 1 & 2
* The third output is expected to be the XOR of inputs 1 & 2
* Yeah that's probably an overkill explanation but the test cases get more complicated
*
* * The number of hidden layers is not required to be 1
*/
void Net_Tester::setup_AND_OR_XOR()
{
    //Allocate memory for expected inputs/outputs. (2-B-3) network
    int num_cases = 4;
    double** inputs = new double*[num_cases];
    double** outputs = new double*[num_cases];
    for(int i = 0; i < 4; i++)
    {
        inputs[i] = new double[2];
        outputs[i] = new double[3];
    }


    //Initialize values in expected input/outputs
    Net_Tester::setup_inputs(inputs, num_cases, 2);
    //Set output cases -> AND, OR, XOR (all of only 2 inputs)
    for(int i = 0; i < num_cases; i++)
    {
        outputs[i][0] = inputs[i][0] && inputs[i][1];
        outputs[i][1] = inputs[i][0] || inputs[i][1];
        outputs[i][2] = (inputs[i][0] || inputs[i][1]) && (inputs[i][0] != inputs[i][1]);
    }


    //Pass expected inputs/outputs to be deep copied
    this->net->set_expected_inputs(inputs, num_cases);
    this->net->set_expected_outputs(outputs, num_cases);


    //Free memory allocated to expected inputs/outputs
    for(int i = 0; i < num_cases; i++)
    {
        delete [] inputs[i];
        delete [] outputs[i];
    }
    delete [] inputs;
    delete [] outputs;
}


void Net_Tester::setup_ABC_tester1()
{
    //Allocated memory for expected input/output arrays (3-B-4 network)
    int num_cases = 8;
    double** inputs = new double*[num_cases];
    double** outputs = new double*[num_cases];
    for(int i = 0; i < num_cases; i++)
    {
        inputs[i] = new double[3];
        outputs[i] = new double[4];
    }
    
    //Initialize values in expected input/outputs
    Net_Tester::setup_inputs(inputs, num_cases, 3);
    for(int i = 0; i < num_cases; i++)
    {
        //First output stores XOR of first two inputs
        outputs[i][0] = (inputs[i][0] || inputs[i][1]) && (inputs[i][0] != inputs[i][1]);
        //Second output stores AND of first ouput with 3rd input
        outputs[i][1] = outputs[i][0] && inputs[i][2];
        //Third output stores OR of first output with 3rd input
        outputs[i][2] = outputs[i][0] && inputs[i][2];
        //Fourth output stores XOR of first output with 3rd input
        outputs[i][3] = (outputs[i][0] || inputs[i][2]) && (outputs[i][0] != inputs[i][2]);
    }

    //Pass expected inputs/outputs to be deep copied
    this->net->set_expected_inputs(inputs, num_cases);
    this->net->set_expected_outputs(outputs, num_cases);


    //Free the memory allocated to the arrays of cases;
    for(int i = 0; i < num_cases; i++)
    {
        delete [] inputs[i];
        delete [] outputs[i];
    }
    delete [] inputs;
    delete [] outputs;
}


void Net_Tester::setup_ABC_tester2()
{
    //Allocated memory for expected input/output arrays (4-B-3 network)
    int num_cases = 16;
    double** inputs = new double*[num_cases];
    double** outputs = new double*[num_cases];
    for(int i = 0; i < num_cases; i++)
    {
        inputs[i] = new double[4];
        outputs[i] = new double[3];
    }
    
    //Initialize values in expected input/outputs
    Net_Tester::setup_inputs(inputs, num_cases, 4);
    for(int i = 0; i < num_cases; i++)
    {
        //First output stores XOR of first two inputs
        outputs[i][0] = (inputs[i][0] || inputs[i][1]) && (inputs[i][0] != inputs[i][1]);
        //Second output stores OR of 3rd and 4th inputs
        outputs[i][1] = inputs[i][2] || inputs[i][3];
        //Third output stores AND of first output and second output
        outputs[i][2] = outputs[i][0] && outputs[i][1];
    }

    //Pass expected inputs/outputs to be deep copied
    this->net->set_expected_inputs(inputs, num_cases);
    this->net->set_expected_outputs(outputs, num_cases);


    //Free the memory allocated to the arrays of cases;
    for(int i = 0; i < num_cases; i++)
    {
        delete [] inputs[i];
        delete [] outputs[i];
    }
    delete [] inputs;
    delete [] outputs;
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
    /*
    if(this->net != nullptr)
    {
        delete this->net;
    }
    */
}


void Net_Tester::test_AND()
{
    Net_Tester::setup_AND();
    cout << endl << endl;
    cout << endl << "————————————————————————" << endl << "Testing AND" << endl;
    cout << "————————————————————————" << endl;
    this->net->train_network(100000, 0.01, 0.3, 0.1, 1.5, 1000);
    cout << "————————————————————————" << endl << endl << endl;
    this->net->run_network(1, 1);
    cout << endl << "————————————————————————" << endl << "End Testing AND" << endl;
    cout << "————————————————————————" << endl;
    cout << endl << endl;
}


void Net_Tester::test_OR()
{
    Net_Tester::setup_OR();
    cout << endl << endl;
    cout << endl << "————————————————————————" << endl << "Testing OR" << endl;
    cout << "————————————————————————" << endl;
    this->net->train_network(100000, 0.01, 0.3, 0.1, 1.5, 1000);
    cout << "————————————————————————" << endl << endl << endl;
    this->net->run_network(1, 1);
    cout << endl << "————————————————————————" << endl << "End Testing OR" << endl;
    cout << "————————————————————————" << endl;
    cout << endl << endl;
}


void Net_Tester::test_XOR()
{
    Net_Tester::setup_XOR();
    cout << endl << endl;
    cout << endl << "————————————————————————" << endl << "Testing XOR" << endl;
    cout << "————————————————————————" << endl;
    this->net->train_network(100000, 0.01, 0.3, 0.1, 1.5, 1000);
    cout << "————————————————————————" << endl << endl << endl;
    this->net->run_network(1, 1);
    cout << endl << "————————————————————————" << endl << "End Testing XOR" << endl;
    cout << "————————————————————————" << endl;
    cout << endl << endl;
}


void Net_Tester::test_AND_OR_XOR()
{
    Net_Tester::setup_AND_OR_XOR();
    cout << endl << endl;
    cout << endl << "————————————————————————" << endl << "Testing AND_OR_XOR" << endl;
    cout << "————————————————————————" << endl;
    this->net->train_network(100000, 0.01, 0.3, 0.1, 1.5, 1000);
    cout << "————————————————————————" << endl << endl << endl;
    this->net->run_network(1, 1);
    cout << endl << "————————————————————————" << endl << "End Testing AND_OR_XOR" << endl;
    cout << "————————————————————————" << endl;
    cout << endl << endl;
}


void Net_Tester::test_ABC_tester1()
{
    Net_Tester::setup_ABC_tester1();
    cout << endl << endl;
    cout << endl << "————————————————————————" << endl << "Testing ABC_tester1" << endl;
    cout << "————————————————————————" << endl;
    this->net->train_network(100000, 0.01, 0.3, 0.1, 1.5, 1000);
    cout << "————————————————————————" << endl << endl << endl;
    this->net->run_network(1, 1);
    cout << endl << "————————————————————————" << endl << "End Testing ABC_tester1" << endl;
    cout << "————————————————————————" << endl;
    cout << endl << endl;
}


void Net_Tester::test_ABC_tester2()
{
    Net_Tester::setup_ABC_tester2();
    cout << endl << endl;
    cout << endl << "————————————————————————" << endl << "Testing ABC_tester2" << endl;
    cout << "————————————————————————" << endl;
    this->net->train_network(100000, 0.01, 0.3, -1.1, 1.5, 1000);
    cout << "————————————————————————" << endl << endl << endl;
    this->net->run_network(1, 1);
    cout << endl << "————————————————————————" << endl << "End Testing ABC_tester2" << endl;
    cout << "————————————————————————" << endl;
    cout << endl << endl;
}