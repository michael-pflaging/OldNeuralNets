#include "neural_net.h"
#include "net_tester.h"

int main()
{
    srand(time(NULL));

    Neural_Net net(2, 3, 3, 1, 4, 0.3, "output", false, 100000, 0.01, 0.1, 1.5, 100);
    
    Net_Tester t(&net);
    t.test_AND();
    //t.test_OR();
    
    return 0;
}