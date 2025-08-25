#include "neural_net.h"
#include <chrono>
#include <ctime> 
#include <cmath>

using namespace std;

double Neural_Net::generate_random_number(int min_rand, int max_rand) const
{
    return ((double)rand() / RAND_MAX) * (max_rand - min_rand) + min_rand;
}


double Neural_Net::sigmoid(double input) const
{
    return 1.0 / (1.0 + std::exp(-input));
}


double Neural_Net::sigmoid_deriv(double input) const
{
    return Neural_Net::sigmoid(input) * (1.0 - Neural_Net::sigmoid(input));
}


double Neural_Net::reLU(double input) const
{
    if(input < 0.0)
        return 0.0;
    else
        return (double)input;
}

double Neural_Net::reLU_deriv(double input) const
{
    if(input > 0.0)
        return 1.0;
    else
        return 0.0;
}


double Neural_Net::calculate_error(double output, double expected) const
{
    return (0.5 * ((expected - output) * (expected - output)));
}


void Neural_Net::set_random_weights(int min_rand, int max_rand)
{
    //Loop through each layer
    for(int n = 0; n < this->num_layers - 1; n++)
    {
        //Loop through each activation's weights in each layer
        for (int k = 0; k < this->num_per_layer[n]; k++)
        {
            //Loop through each weight for each activation
            for (int j = 0; j < this->num_per_layer[n + 1]; j++)
            {
               //Assign a random value within the specified range to this weight
               this->weights[n][k][j] = Neural_Net::generate_random_number(min_rand, max_rand);
            }
        }
    }
}


void Neural_Net::save_weights()
{
    //blah
}


void Neural_Net::allocate_training_memory()
{
    this->raw_vals = new double*[this->num_layers];
    this->error_caused = new double*[this->num_layers];
    this->actual_vals = new double*[this->num_layers];

    actual_vals[0] = new double[this->num_per_layer[0]];
    for(int i = 1; i < this->num_layers; i++)
    {
        raw_vals[i] = new double[this->num_per_layer[i]];
        error_caused[i] = new double[this->num_per_layer[i]];
        actual_vals[i] = new double[this->num_per_layer[i]];
    }
}


void Neural_Net::free_training_memory()
{
   delete [] this->actual_vals[0];
    for(int i = 1; i < this->num_layers; i++)
    {
        delete [] this->raw_vals[i];
        delete [] this->error_caused[i];
        delete [] this->actual_vals[i];
    }
    delete [] this->raw_vals;
    delete [] this->error_caused;
    delete [] this->actual_vals;
}


double Neural_Net::activation_func(double input) const
{
    return Neural_Net::sigmoid(input);
}


double Neural_Net::activation_func_deriv(double input) const
{
    return Neural_Net::sigmoid_deriv(input);
}


Neural_Net::Neural_Net()
{
    this->activations = nullptr;
    this->weights = nullptr;
    this->num_per_layer = nullptr;
    this->num_layers = 0;

    this->Ti = nullptr;
    this->raw_vals = nullptr;
    this->actual_vals = nullptr;
    this->error_caused = nullptr;
    
    this->single_set_error = 0.0;
    this->expected_input_values = nullptr;
    this->expected_output_values = nullptr;
    this->num_input_cases = 0;
    this->num_output_cases = 0;
    this->weight_file_name = "";

    this->single_set_error = 0.0;
}


Neural_Net::Neural_Net(int* num_per_layer, int num_layers, string weight_file_name)
{
    this->num_layers = num_layers;
    this->num_per_layer = new int[this->num_layers];
    for(int i = 0; i < this-> num_layers; i++)
        this->num_per_layer[i] = num_per_layer[i];

    this->single_set_error = 0.0;


    this->Ti = nullptr;
    this->raw_vals = nullptr;
    this->actual_vals = nullptr;
    this->error_caused = nullptr;

    this->expected_input_values = nullptr;
    this->expected_output_values = nullptr;
    this->num_input_cases = 0;
    this->num_output_cases = 0;

    this->weight_file_name = weight_file_name;

    //Allocate arrays
    this->activations = new double*[this->num_layers];
    for(int n = 0; n < this->num_layers; n++)
    {
        this->activations[n] = new double[this->num_per_layer[n]];
    }
    
    //Allocate weights
    this->weights = new double**[this->num_layers - 1];
    for(int n = 0; n < this->num_layers - 1; n++)
    {
        this->weights[n] = new double*[this->num_per_layer[n]];
        for(int j = 0; j < this->num_per_layer[n]; j++)
        {
            this->weights[n][j] = new double[this->num_per_layer[n+1]];
        }
    }
}


Neural_Net::~Neural_Net()
{
    //Activations 2-D array
    if(activations != nullptr)
    {
        for(int n = 0; n < this->num_layers; n++)
        {
            if(this->activations[n] != nullptr)
            {
                delete [] this->activations[n];
            }
        }
        delete [] this->activations;
    }
    

    //Weights 3-D array
    
    if(this->weights != nullptr)
    {
        for(int n = 0; n < this->num_layers - 1; n++)
        {
            if(this->weights[n] != nullptr)
            {
                for(int j = 0; j < this->num_per_layer[n]; j++)
                {
                    delete [] this->weights[n][j];
                }
                delete [] this->weights[n];
            }
        }
        delete[] this->weights;
    }
    
    
    if(this->expected_input_values != nullptr)
    {
        for(int i = 0; i < this->num_input_cases; i++)
        {
            if(this->expected_input_values[i] != nullptr) 
            {
                delete [] this->expected_input_values[i];
            }
        }
        delete [] this->expected_input_values;
    }

    if(this->expected_output_values != nullptr)
    {
        for(int i = 0; i < this->num_output_cases; i++)
        {
            if(this->expected_output_values[i] != nullptr) 
            {
                delete [] this->expected_output_values[i];
            }
        }
        delete [] this->expected_output_values;
    }
    
    delete [] this->num_per_layer;
}


double Neural_Net::run_network(int print_table, int print_error)
{
    double total_error = 0.0;

    //Iterate through the different test cases of different input activations
    for(int a = 0; a < this->num_input_cases; a++)
    {
        //Assign the input activation values
        for(int m = 0; m < this->num_per_layer[0]; m++)
        {
            this->activations[0][m] = this->expected_input_values[a][m];
        }

        //Calculate values for every hidden layer and output layer
        for(int n = 1; n < this->num_layers; n++)
        {
            //This counter "c" keeps track of the current neuron in current layer
            for(int c = 0; c < this->num_per_layer[n]; c++)
            {
                //Zero the value of the current neuron
                this->activations[n][c] = 0.0;

                //Loop through each neuron in the previous layer
                for(int p = 0; p < this->num_per_layer[n-1]; p++)
                {
                    //Sum the dot products of the previous layer's activation & weight
                    this->activations[n][c] += this->activations[n-1][p] * this->weights[n-1][p][c];
                }
                //Then apply activation function
                this->activations[n][c] = Neural_Net::activation_func(this->activations[n][c]);
            }
        }


        double this_error = 0.0;
        //Calculate error by summing the error from each output for this case
        for(int i = 0; i < this->num_per_layer[this->num_layers - 1]; i++)
        {
            this_error += Neural_Net::calculate_error(this->activations[this->num_layers-1][i], this->expected_output_values[a][i]);
        }
        //Add to running total
        total_error += this_error;

        //Print truth table if told to
        if(print_table)
        {
            //Print out the input activations
            for(int k = 0; k < this->num_per_layer[0]; k++)
                cout << this->activations[0][k] << " ";
            cout << "  ";

            //Print expected outputs
            for(int i = 0; i < this->num_per_layer[3]; i++)
               cout << this->expected_output_values[a][i] << " ";
            cout << "  ";

            //Print out the actual output
            for (int i = 0; i < this->num_per_layer[this->num_layers - 1]; i++)
               cout << this->activations[this->num_layers - 1][i] << "\t";

            //Print individual error if told to
            if(print_error)
                cout << this_error;
            cout << endl;
        }
    } //for (int a = 0; a < net.expectedInputValues.length; a++)

    //Print total error if told to
    if (print_error)
        cout << "Total Error: " << total_error << endl;
    
    return total_error;
}


double Neural_Net::run_for_training(double error, int which_case)
{
    //Do forward prop and fill arrays starting from first hidden layer
    for(int l = 1; l < this->num_layers; l++)
    {
        //For each neuron in the layer
        for(int c = 0; c < this->num_per_layer[l]; c++)
        {
            //Zero raw value
            this->raw_vals[l][c] = 0.0;
            //For each neuron in the previous layer, sum dot products to calculate raw value for current neuron
            for(int p = 0; p < this->num_per_layer[l-1]; p++)
                this->raw_vals[l][c] += this->actual_vals[l-1][p] * this->weights[l-1][p][c];
            //Now store actual value of the raw value calculated
            this->actual_vals[l][c] = Neural_Net::activation_func(this->raw_vals[l][c]);
        }
    }

    //For each output neuron
    for(int i = 0; i < this->num_per_layer[this->num_layers-1]; i++)
    {
        /*
        * Error Function is 1/2 * (Expected output - acutal output)^2, so the partial of the error with respect
        * to the output activation is (expected output - actual output)
        * Given the output activation, the partial of the activation with respect to its raw input is the
        * derivative of the activation function of the raw value of the activation
        * Multiplied together, this makes the error_caused variable store the partial of the error with respect to the
        * raw value of the activation
        */
        this->error_caused[this->num_layers-1][i] = (this->Ti[i] - this->actual_vals[this->num_layers-1][i]) 
                                                    * Neural_Net::activation_func_deriv(this->raw_vals[this->num_layers-1][i]);
        //Sum up single set error
        this->single_set_error += Neural_Net::calculate_error(this->actual_vals[this->num_layers-1][i], this->Ti[i]);
    }

    //If all single set errors have been calculated, then new total error of the network has been calculated
    if(which_case == this->num_output_cases - 1)
    {
        error = this->single_set_error;
        this->single_set_error = 0.0;
        return error;
    } //if (whichCase == this->expectedOutputValues.length-1)
    //Otherwise new total error has not been calculated yet and return previous error
    else
    {
        return error;
    } //else
} //private static double runForTraining(NeuralNetwork net, double error, int whichCase)


void Neural_Net::train_network(int max_cycles, double min_error, double lambda,
                                double min_rand, double max_rand, int save_frequency)
{
    Neural_Net::set_random_weights(min_rand, max_rand);
    Neural_Net::allocate_training_memory();

    //Start timer
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    //Error calculated by running network, initial value allows loop start
    double error = min_error;
    //Counter for # of cycles
    int cycles = 0;

    //Loop through training until a end condition is met
    while(cycles < max_cycles && error >= min_error)
    {
        //Cycle what is expected from the output cases
        int this_case = cycles % this->num_output_cases;
        this->Ti = this->expected_output_values[this_case];

        //Initialize input vals
        for(int i = 0; i < this->num_per_layer[0]; i++)
            this->actual_vals[0][i] = this->expected_input_values[this_case][i];

        //Fill forward prop arrays, calculate error (is only adjusted once all single set errors are calculated)
        error = Neural_Net::run_for_training(error, this_case);

        //Iterating from 2nd to last layer (last weight layer) to 1st layer - All hidden layers
        for(int l = (this->num_layers - 2); l > 0; l--)
        {
            //For each neuron in current layer -> c represents current neuron
            for(int c = 0; c < this->num_per_layer[l]; c++)
            {
                this->error_caused[l][c] = 0.0;
                //For each neuron in next layer -> n represents next neuron
                for(int n = 0; n < this->num_per_layer[l+1]; n++)
                {
                    this->error_caused[l][c] += this->error_caused[l+1][n] * this->weights[l][c][n];

                    /*
                    * This ends up being lambda * (negative partial of error with respect to weight)
                    * error_caused is the partial of error with respect to the raw value of the activation
                    * specified by [l+1][n], and actual vals is the value post-activation function of the activation
                    * specified by [l][c]. 
                    * The weight being updated -> [l][c][n] is the weight that connects these two activations
                    */
                    this->weights[l][c][n] += lambda * this->actual_vals[l][c] * this->error_caused[l+1][n];
                }
                this->error_caused[l][c] = this->error_caused[l][c] * Neural_Net::activation_func_deriv(this->raw_vals[l][c]);
            }
        }
        //0th layer - input activations
        for(int c = 0; c < this->num_per_layer[0]; c++)
        {
            for(int n = 0; n < this->num_per_layer[1]; n++)
            {
                this->weights[0][c][n] += lambda * this->actual_vals[0][c] * this->error_caused[1][n];
            }
        }
        

        //Save weights every saveFrequency cycles
        if(cycles % save_frequency == 0)
        {
            Neural_Net::save_weights();
        }
         
        //Increment # of cycles
        cycles++;
    } //while (cycles < cyclesMax && Error > minError)

    //End timer
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();

    //Get white space
    cout << endl << endl;

    //Print why it stopped
    if(cycles >= max_cycles)
    {
        cout << "Hit maximum number of loops." << endl;
    } //if (cycles >= this->cyclesMax)
    else
    {
        cout << "Error was smaller than minError." << endl;
    } //else

    //Print config and rand range
    cout << "Config: " << this->num_per_layer[0] << " " << this->num_per_layer[1] << " " << this->num_per_layer[2] << " " << this->num_per_layer[3] << endl;
    cout << "Rand range: " << min_rand << " - " << max_rand << endl;

    //Print N max, Emin, and lambda
    cout << "N max was: " << max_cycles << endl;
    cout << "Min Error was " << min_error << endl;
    cout << "Lambda was: " << lambda << endl;

    //Print Error then # of iteration, using runNetwork to get total error as the error used to stop is almost but not quite the same
    cout << "Error was: " << Neural_Net::run_network(0, 0) << endl;
    cout << "# of iterations: " << cycles << endl;

    //Now print truth table
    cout << "Truth Table" << endl;
    Neural_Net::run_network(1, 0);

    std::chrono::duration<double> elapsed_seconds = end-start;
    //Print time taken
    double time_taken = elapsed_seconds.count();
    int min = (int)(time_taken/60);

    if(time_taken > 60)
    {
        cout << "Time Taken: " << min << "m " << time_taken - (min * 60) << "s" << endl;
    } //if (timeTaken > 60)
    else
    {
        cout << "Time Taken: " << time_taken << "s" << endl;
    }

    //Following whitespace
    cout << endl << endl;

    //Save weights
    Neural_Net::save_weights();
    Neural_Net::free_training_memory();

} //private static void trainNetwork(NeuralNetwork net, int saveFrequency)


void Neural_Net::set_expected_inputs(double** inputs, int n)
{
    if(this->expected_input_values != nullptr)
    {
        for(int i = 0; i < this->num_input_cases; i++)
        {
            if(this->expected_input_values[i] != nullptr) 
            {
                delete [] this->expected_input_values[i];
            }
        }
        delete [] this->expected_input_values;
    }

    this->expected_input_values = new double*[n];
    for(int i = 0; i < n; i++)
    {
        this->expected_input_values[i] = new double[this->num_per_layer[0]];
        for(int j = 0; j < this->num_per_layer[0]; j++)
        {
            this->expected_input_values[i][j] = inputs[i][j];
        }
    }
    this->num_input_cases = n;
}


void Neural_Net::set_expected_outputs(double** outputs, int n)
{
    //Free already-allocated memory, if applicable
    if(this->expected_output_values != nullptr)
    {
        for(int i = 0; i < this->num_output_cases; i++)
        {
            if(this->expected_output_values[i] != nullptr) 
            {
                delete [] this->expected_output_values[i];
            }
        }
        delete [] this->expected_output_values;
    }
    
    //Perform a deep copy of the expected outputs given
    this->expected_output_values = new double*[n];
    for(int i = 0; i < n; i++)
    {
        this->expected_output_values[i] = new double[this->num_per_layer[this->num_layers - 1]];
        for(int j = 0; j < this->num_per_layer[this->num_layers - 1]; j++)
        {
            this->expected_output_values[i][j] = outputs[i][j];
        }
    }
    this->num_output_cases = n;
}