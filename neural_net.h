#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <iostream>

using namespace std;

class Neural_Net
{
    
private:
    //Activations
    double** activations;
    //Weights
    double*** weights;
    //Inputs, Hidden activations, Outputs, Layers, Learning Factor
    int num_inputs;
    int num_hidden1;
    int num_hidden2;
    int num_outputs;
    int num_layers;
    double λ;
    //Training arrays
    double* am;
    double* Θk;
    double* ak;
    double* Θj;
    double* aj;
    double* Ψj;
    double* ai;
    double* ψi;
    double* Ti;
    //For storing random ranges
    double min_rand;
    double max_rand;
    //Training conditions
    double min_error;
    int cycles_max;
    int save_frequency;
    //For storing sum of single set errors
    double single_set_error;
    //For the possible inputs and expected outputs
    double** possible_input_values;
    double** expected_output_values;
    int num_input_cases;
    int num_output_cases;

    string weight_file_name;
    //True makes the network train, false makes the network run
    bool train_or_run;

    //For initialization / deletion purposes, this leads to excess memory usage in activations and weights array since
    //They should be jagged arrays where needed to cut down on memory use but I am lazy
    int max_activations;

    /**
    * Generate a random number from the member variables min_rand to max_rand
    * @return a double that is greater than or equal to the min value, and less than the max.
    */
    double generate_random_number() const;


    /**
    * Calculate and return the result of the sigmoid function 
    * @param input the value being plugged in to the sigmoid function
    * @return the value of of the sigmoid after x has been plugged in
    */
    double sigmoid(double input) const;


    /**
    * Calculate and return the value of the derivative of the sigmoid function
    * @param input the value passed to the derivate function
    * @return the value of the deriv of the sigmoid using the value of the sigmoid
    */
    double sigmoid_deriv(double input) const;


    /**
    * Calculate and return the value of the error given a certain ouptut value and
    * expected value
    * @param output the output value being compared to the expected value
    * @param expected the expected output value of the neural net
    * @return the calculated error value
    */
    double calculate_error(double output, double expected) const;


    /**
    * Set the weights of the neural network to be random values that are between
    * the member variables min_rand and max_rand. Technically this is
    * not the most efficient since every space in the weight array is filled regardless
    * of if it is needed
    */
    void set_random_weights();

    /**
    * Write the contents of this network to a json file using JSON.simple
    * member variable weight_file_name is the destination to save the weights
    */
    void save_weights();

public:
    //Default constructor
    Neural_Net();

    /**
    * Parametrized constructor that initializes the network
    * @param num_inputs the number of input activations
    * @param num_hidden1 the number of hidden activations in the 2nd layer
    * @param num_hidden2 the number of hidden activations in the 3rd layer
    * @param num_outputs the number of output activations
    * @param num_layers the number of layers
    * @param λ the learning factor
    * @param weight_file_name the name of the weight file to save/load from
    * @param train_or_run boolean whether training or running
    * @param cycles_max the maximum number of cycles while training
    * @param min_error the minimum error at which training stops
    * @param min_rand the lower bound of the random weights
    * @param max_rand the upper bound of the random weights
    * @param save_frequency how often weights are saved while training
    */
    Neural_Net(int num_inputs, int num_hidden1, int num_hidden2, int num_outputs, int num_layers, double λ, string weight_file_name, 
                        bool train_or_run, int cycles_max, double min_error, double min_rand, double max_rand, int save_frequency);


    //Destructor
    ~Neural_Net();


    /**
    * Run the network
    * @param print_complete boolean determining whether all info is printed
    * @param print_table boolean determining if truth table is printed
    * @return the total error of the network
    */
    double run_network(bool print_complete, bool print_table);


    /**
    * Run the forward prop part of the network for training
    * @param error the current error of the network
    * @param which_case the current single set case being trained
    * @return the single set error of the network
    */
    double run_for_training(double error, int which_case);


    /**
    * Train the neural network using gradient descent with backpropagation implemented for efficiency
    */
    void train_network();


    //Setter for possible inputs
    void set_possible_inputs(double**, int);

    //Setter for expected outputs
    void set_expected_outputs(double**, int);
};

#endif