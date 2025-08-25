#include "neural_net.h"
#include <chrono>
#include <ctime> 

double Neural_Net::generate_random_number() const
{
    return ((double)rand() / RAND_MAX) * (this->max_rand - this->min_rand) + this->min_rand;
}


double Neural_Net::sigmoid(double input) const
{
    return 1.0 / (1.0 + exp(-input));
}


double Neural_Net::sigmoid_deriv(double input) const
{
    return Neural_Net::sigmoid(input) * (1.0 - Neural_Net::sigmoid(input));
}


double Neural_Net::calculate_error(double output, double expected) const
{
    return (0.5 * ((expected - output) * (expected - output)));
}


void Neural_Net::set_random_weights()
{
    //Loop through each layer
    for(int n = 0; n < this->num_layers - 1; n++)
    {
        //Loop through each activation's weights in each layer
        for (int k = 0; k < this->max_activations; k++)
        {
            //Loop through each weight for each activation
            for (int j = 0; j < this->max_activations; j++)
            {
               //Assign a random value within the specified range to this weight
               this->weights[n][k][j] = Neural_Net::generate_random_number();
            }
        }
    }
}


void Neural_Net::save_weights()
{
    //blah
}


Neural_Net::Neural_Net()
{
    this->activations = nullptr;
    this->weights = nullptr;
    this->num_inputs = 0;
    this->num_hidden1 = 0;
    this->num_hidden2 = 0;
    this->num_outputs = 0;
    this->num_layers = 0;
    this->λ = 0.0;
    this->am = nullptr;
    this->Θk = nullptr;
    this->ak = nullptr;
    this->Θj = nullptr;
    this->aj = nullptr;
    this->Ψj = nullptr;
    this->ai = nullptr;
    this->ψi = nullptr;
    this->Ti = nullptr;
    this->min_rand = 0.0;
    this->max_rand = 0.0;
    this->min_error = 0.0;
    this->cycles_max = 0;
    this->save_frequency = 0;
    this->single_set_error = 0.0;
    this->possible_input_values = nullptr;
    this->expected_output_values = nullptr;
    this->num_input_cases = 0;
    this->num_output_cases = 0;
    this->weight_file_name = "";
    //Default is run
    this->train_or_run = false;
    this->max_activations = 0;
}


Neural_Net::Neural_Net(int num_inputs, int num_hidden1, int num_hidden2, int num_outputs, int num_layers, double λ, string weight_file_name, 
                        bool train_or_run, int cycles_max, double min_error, double min_rand, double max_rand, int save_frequency)
{
    this->num_inputs = num_inputs;
    this->num_hidden1 = num_hidden1;
    this->num_hidden2 = num_hidden2;
    this->num_outputs = num_outputs;
    this->num_layers = num_layers;
    this->λ = λ;
    this->am = new double[this->num_inputs];
    this->Θk = new double[this->num_hidden1];
    this->ak = new double[this->num_hidden1];
    this->Θj = new double[this->num_hidden2];
    this->aj = new double[this->num_hidden2];
    this->Ψj = new double[this->num_hidden2];
    this->ai = new double[this->num_outputs];
    this->ψi = new double[this->num_outputs];
    this->Ti = new double[this->num_outputs];
    this->weight_file_name = weight_file_name;
    this->train_or_run = train_or_run;
    this->cycles_max = cycles_max;
    this->min_error = min_error;
    this->min_rand = min_rand;
    this->max_rand = max_rand;
    this->save_frequency = save_frequency;

    //Set max activations
    this->max_activations = this->num_inputs;
    if(this->num_hidden1 > this->max_activations)
        this->max_activations = this->num_hidden1;
    if(this->num_hidden2 > this->max_activations)
        this->max_activations = this->num_hidden2;
    if(this->num_outputs > this->max_activations)
        this->max_activations = this->num_outputs;

    //Allocate arrays
    this->activations = new double*[this->num_layers];
    for(int n = 0; n < this->num_layers; n++)
    {
        this->activations[n] = new double[this->max_activations];
    }
    
    //Allocate weights
    this->weights = new double**[this->num_layers - 1];
    for(int n = 0; n < this->num_layers -1; n++)
    {
        this->weights[n] = new double*[this->max_activations];
        for(int j = 0; j < this->max_activations; j++)
        {
            this->weights[n][j] = new double[this->max_activations];
        }
    }
}


Neural_Net::~Neural_Net()
{
    //Activations 2-D array
    if(this->activations != nullptr)
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
                for(int j = 0; j < this->max_activations; j++)
                {
                    delete [] this->weights[n][j];
                }
                delete [] this->weights[n];
            }
        }
        delete[] this->weights;
    }

    //Training 1-D arrays
    if(this->am != nullptr) {delete this->am;}
    if(this->Θk != nullptr) {delete this->Θk;}
    if(this->ak != nullptr) {delete this->ak;}
    if(this->Θj != nullptr) {delete this->Θj;}
    if(this->aj != nullptr) {delete this->aj;}
    if(this->Ψj != nullptr) {delete this->Ψj;}
    if(this->ai != nullptr) {delete this->ai;}
    if(this->ψi != nullptr) {delete this->ψi;}
    if(this->Ti != nullptr) {delete this->Ti;}

    if(this->possible_input_values != nullptr)
    {
        for(int i = 0; i < this->num_input_cases; i++)
        {
            if(this->possible_input_values[i] != nullptr) 
            {
                delete [] this->possible_input_values[i];
            }
        }
        delete [] this->possible_input_values;
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
}


double Neural_Net::run_network(bool print_complete, bool print_table)
{
    double total_error = 0.0;

    //Iterate through the different test cases of different input activations
    for(int a = 0; a < this->num_input_cases; a++)
    {
        //Assign the input activation values
        for(int m = 0; m < num_inputs; m++)
        {
            this->activations[0][m] = this->possible_input_values[a][m];
        }

        /*
        * Evaluate the first hidden layer (n = 1)
        * Iterate through each neuron in the hidden activation layer
        */
        for (int k = 0; k < this->num_hidden1; k++)
        {
            //Declare Θk
            double Θk = 0.0;

            //Loop through input activations
            for (int m = 0; m < this->num_inputs; m++)
            {
               //Sum the dot products into the value of the activation
               Θk += this->activations[0][m] * this->weights[0][m][k];
            } //for (int m = 0; m < net.numInputs; m++)

            //Then do the sigmoid of Θk to get the value of the activation
            this->activations[1][k] = Neural_Net::sigmoid(Θk);
        } //for (int k = 0; k < net.numHidden1; k++)

        /*
        * Evaluate the second hidden layer (n = 2)
        * Iterate through each neuron in the hidden activation layer
        */
        for(int j = 0; j < this->num_hidden2; j++)
        {
            //Declare Θj
            double Θj = 0.0;

            //Loop through hidden activations from previous layer (n=1)
            for (int k = 0; k < this->num_hidden1; k++)
            {
               //Sum the dot products into the value of the activation
               Θj += this->activations[1][k] * this->weights[1][k][j];
            } //for (int k = 0; k < net.numHidden1; k++)

            //Then do the sigmoid of Θj to get the value of the activation
            this->activations[2][j] = Neural_Net::sigmoid(Θj);
        } //for (int j = 0; j < net.numHidden2; j++)

        //Now calculate output(s), loop through number of outputs
        for (int i = 0; i < this->num_outputs; i++)
        {
            //Declare Θi
            double Θi = 0.0;

            for (int j = 0; j < this->num_hidden2; j++)
            {
               //Sum the dot products into the value of the activation
               Θi += this->activations[2][j] * this->weights[2][j][i];
            } //for (int j = 0; j < net.numHidden2; j++)

            //Then do the sigmoid
            this->activations[3][i] = Neural_Net::sigmoid(Θi);
         } //for (int i = 0; i < net.numOutputs; i++)
         
        //Calculate error
        double this_error = 0.0;
        for(int i = 0; i < this->num_outputs; i++)
        {
            //Add error for this run to total error
            this_error += Neural_Net::calculate_error(this->activations[this->num_layers-1][i], this->expected_output_values[a][i]);
            total_error += this_error;
        } //for (int i = 0; i < net.numOutputs; i++)

        //Print truth table if required
        if (print_table)
        {
            //Print out the input activations
            for (int k = 0; k < this->num_inputs; k++)
            {
               cout << this->activations[0][k] << " ";
            } //for (int k = 0; k < net.numInputs; k++)

            //Get a bit more space
            cout << "  ";;

            //Print out the expected outputs
            for (int i = 0; i < this->num_outputs; i++)
            {
               cout << this->expected_output_values[a][i] << " ";
            } //for (int i = 0; i < net.numOutputs; i++)

            //Get a bit more space
            cout << "  ";;

            //Print out the actual outputs - hardcoded for 4 layer network
            for (int i = 0; i < this->num_outputs; i++)
            {
               cout << this->activations[3][i] << "\t";
            } //for (int i = 0; i < net.numOutputs; i++)

            //Now go to a new line
            cout << endl;
        } //if (printTable)
      
        //Truth table but with error
        if (print_complete)
        {
            //Print out the input activations
            for (int k = 0; k < this->num_inputs; k++)
            {
               cout << this->activations[0][k] << " ";
            } //for (int k = 0; k < net.numInputs; k++)
         
            //Get a bit more space
            cout << "  ";;
         
            //Print out the expected outputs
            for (int i = 0; i < this->num_outputs; i++)
            {
               cout << this->expected_output_values[a][i] << " ";
            } //for (int i = 0; i < net.numOutputs; i++)
         
            //Get a bit more space
            cout << "  ";;
         
            //Print out the actual outputs - hardcoded for 4 layer network
            for (int i = 0; i < this->num_outputs; i++)
            {
               cout << this->activations[3][i] << "\t";
            } //for (int i = 0; i < net.numOutputs; i++)
         
            //Print error
            cout << this_error << endl;
        } //if (printComplete)

    } //for (int a = 0; a < net.possibleInputValues.length; a++)

    if (print_complete)
    {
        cout << "Total Error: " << total_error << endl;
    }
    return total_error;
}


double Neural_Net::run_for_training(double error, int which_case)
{
    //Calculate jk, Θj, ψi. For each output activation
    for(int i = 0; i < this->num_outputs; i++)
    {
        //Zero Θi before calculating it
        double Θi = 0.0;
        //For each hidden activation on layer 3 (n=2)
        for(int j = 0; j < this->num_hidden2; j++)
        {
            //Zero Θj
            this->Θj[j] = 0.0;
            //For each hidden activation on layer 3 (n=1)
            for(int k = 0; k < this->num_hidden1; k++)
            {
                //Zero Θk
                this->Θk[k] = 0.0;
                //For each input activation
                for(int m = 0; m < this->num_inputs; m++)
                {
                    this->Θk[k] += this->am[m] * this->weights[0][m][k];
                } //for (int m = 0; m < this->numInputs; m++)
                //Calculate and store ak and Θj
                this->ak[k] = Neural_Net::sigmoid(this->Θk[k]);
                this->Θj[j] += this->ak[k] * this->weights[1][k][j];
            } //for (int k = 0; k < this->numHidden1; k++)
            //Calculate and store aj
            this->aj[j] = Neural_Net::sigmoid(this->Θj[j]);
            Θi += this->aj[j] * this->weights[2][j][i];
        } //for (int j = 0; j < this->numHidden2; j++)

        this->ai[i] = Neural_Net::sigmoid(Θi);
        //(this->Ti[i] - Fi) is the difference between expected and actual output
        this->ψi[i] = (this->Ti[i] - this->ai[i]) * Neural_Net::sigmoid_deriv(Θi);
        //Sum up single set error
        this->single_set_error += Neural_Net::calculate_error(Neural_Net::sigmoid(Θi), this->Ti[i]);
    } //for (int i = 0; i < this->numOutputs; i++)

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


void Neural_Net::train_network()
{     
    Neural_Net::set_random_weights();

    //start timer
    std::__1::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    //Error calculated by running network - set to 1 now but just temporary until network is run
    double error = 1.0;
    //Number of cycles
    int cycles = 0;
    //To store the single set error

    //Loop through training until a end condition is met
    while(cycles < this->cycles_max && error > this->min_error)
    {
        int this_case = cycles % this->num_output_cases;
        //Cycle what is expected from the 4 output cases
        this->Ti = this->expected_output_values[this_case];
        this->am = this->possible_input_values[this_case];

        //Fill forward prop arrays, calculate error (is only adjusted once all single set errors are calculated)
        error = Neural_Net::run_for_training(error, this_case);

        //Declare Ωj
        double Ωj = 0.0;
        //For each hidden activation on 3rd layer (n=2)
        for(int j = 0; j < this->num_hidden2; j++)
        {
            //Zero Ωj
            Ωj = 0.0;
            //For each output activation
            for(int i = 0; i < this->num_outputs; i++)
            {
                Ωj += this->ψi[i] * this->weights[2][j][i];
                //Update w2xx weights
                this->weights[2][j][i] += this->λ * this->aj[j] * this->ψi[i];
            } //for (int i = 0; i < this->numOutputs; i++)

            this->Ψj[j] = Ωj * Neural_Net::sigmoid_deriv(this->Θj[j]);
        } //for (int j = 0; j < this->numHidden2; j++)

        //Declare Ωk 
        double Ωk = 0.0;
        //For each hidden activation on 2nd layer (n=1)
        for(int k = 0; k < this->num_hidden1; k++)
        {
            //Zero Ωk
            Ωk = 0.0;
            //For each hidden activation on 3rd layer (n=2)
            for(int j = 0; j < this->num_hidden2; j++)
            {
                Ωk += this->Ψj[j] * this->weights[1][k][j];
                this->weights[1][k][j] += this->λ * this->ak[k] * this->Ψj[j];
            } //for (int j = 0; j < this->numHidden2; j++)

            //For each input activation
            for(int m = 0; m < this->num_inputs; m++)
            {
                this->weights[0][m][k] += this->λ * this->am[m] * Ωk * Neural_Net::sigmoid_deriv(this->Θk[k]);
            } //for (int m = 0; m < this->numInputs; m++)
        } //for (int k = 0; k < this->numHidden1; k++)
         

        //Save weights every saveFrequency cycles
        if(cycles % this->save_frequency == 0)
        {
            Neural_Net::save_weights();
        }
         
        //Increment # of cycles
        cycles++;
    } //while (cycles < cyclesMax && Error > minError)

    //End timer
    std::__1::chrono::system_clock::time_point end = std::chrono::system_clock::now();

    //Get white space
    cout << endl << endl;

    //Print why it stopped
    if(cycles >= this->cycles_max)
    {
        cout << "Hit maximum number of loops." << endl;
    } //if (cycles >= this->cyclesMax)
    else
    {
        cout << "Error was smaller than minError." << endl;
    } //else

    //Print config and rand range
    cout << "Config: " << this->num_inputs << " " << this->num_hidden1 << " " << this->num_hidden2 << " " << this->num_outputs << endl;
    cout << "Rand range: " << this->min_rand << " - " << this->max_rand << endl;

    //Print N max, Emin, and lambda
    cout << "N max was: " << this->cycles_max << endl;
    cout << "Min Error was " << this->min_error << endl;
    cout << "λ was: " << this->λ << endl;

    //Print Error then # of iteration, using runNetwork to get total error as the error used to stop is almost but not quite the same
    cout << "Error was: " << Neural_Net::run_network(false, false) << endl;
    cout << "# of iterations: " << cycles << endl;

    //Now print truth table
    cout << "Truth Table" << endl;
    Neural_Net::run_network(false, true);

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

} //private static void trainNetwork(NeuralNetwork net, int saveFrequency)


void Neural_Net::set_possible_inputs(double** inputs, int num)
{
    this->possible_input_values = inputs;
    this->num_input_cases = num;
}

//Setter for possible outputs
void Neural_Net::set_expected_outputs(double** outputs, int num)
{
    this->expected_output_values = outputs;
    this->num_output_cases = num;
}