#include "neural_net.h"
#include "net_tester.h"

int main()
{
    srand(time(NULL));

    int num_layers = 4;
    int* layers = new int[num_layers];


    //First testing block: 2-3-3-1
    layers[0] = 2;
    layers[1] = 3;
    layers[2] = 3;
    layers[3] = 1;
    Neural_Net net1(layers, num_layers, "output");
    Net_Tester t1(&net1);
    t1.test_AND();
    t1.test_OR();
    t1.test_XOR();
    
    
    //Second testing block: 2-3-3-3
    layers[3] = 3;
    Neural_Net net2(layers, num_layers, "output");
    Net_Tester t2(&net2);
    t2.test_AND_OR_XOR();
    
    //Third testing block: 3-5-5-4
    layers[0] = 3;
    layers[1] = 5;
    layers[2] = 5;
    layers[3] = 4;
    Neural_Net net3(layers, num_layers, "output");
    Net_Tester t3(&net3);
    t3.test_ABC_tester1();

    //Fourth testing block: 4-5-5-3
    layers[0] = 4;
    layers[3] = 3;
    Neural_Net net4(layers, num_layers, "output");
    Net_Tester t4(&net4);
    t4.test_ABC_tester2();
    

    delete [] layers;
    
    return 0;
}