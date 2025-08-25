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
    int* num_per_layer;
    int num_layers;
    
    //Training arrays
    double* Ti;
    double** raw_vals;
    double** actual_vals;
    double** error_caused;

    //For storing sum of single set errors
    double single_set_error;
    //For the expected inputs and expected outputs
    double** expected_input_values;
    double** expected_output_values;
    int num_input_cases;
    int num_output_cases;

    string weight_file_name;

    /**
    * Generate a random number from the member variables min_rand to max_rand
    * @return a double that is greater than or equal to the min value, and less than the max.
    */
    double generate_random_number(int min_rand, int max_rand) const;


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

    double reLU(double input) const;
    double reLU_deriv(double input) const;

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
    void set_random_weights(int min_rand, int max_rand);

    /**
    * Write the contents of this network to a json file using JSON.simple
    * member variable weight_file_name is the destination to save the weights
    */
    void save_weights();


    void allocate_training_memory();
    void free_training_memory();

    /*
    * Wrappers for the activation function, allows for easy changes
    * to which activation function is being used in the program
    */
    double activation_func(double input) const;
    double activation_func_deriv(double input) const;


public:
    //Default constructor
    Neural_Net();

    /**
    * Parametrized constructor that initializes the network
    * @param num_per_layer an array storing the number of nodes per layer in the network
    * @param num_layers the number of layers
    * @param weight_file_name the name of the weight file to save/load from
    */
    Neural_Net(int* num_per_layer, int num_layers, string weight_file_name);


    //Destructor
    ~Neural_Net();


    /**
    * Runs the network, assumes that expected input values and # of expected input values
    * have been initialized correctly
    * @param print_table (int) storing boolean determining if truth table is printed
    * @param print_error (int) storing boolean determining if error is printed in truth table
    * @return the total error of the network
    */
    double run_network(int print_table, int print_error);


    /**
    * Run the forward prop part of the network for training
    * @param error the current error of the network
    * @param which_case the current single set case being trained
    * @return the single set error of the network
    */
    double run_for_training(double error, int which_case);


    /**
    * Train the neural network using gradient descent with backpropagation implemented for efficiency
    * Takes in the maximum number of cycles and minimum error as parameters
    */
    void train_network(int max_cycles, double min_error, double lambda, 
                    double min_rand, double max_rand, int save_frequency);


    //Performs a deep copy to set expected inputs. n = # of cases
    void set_expected_inputs(double** inputs, int n);

    //Performs a deep copy to set expected outputs. n = # of cases
    void set_expected_outputs(double** outputs, int n);
};

#endif