#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <cstdlib>
#include <time.h>

using namespace std;

double dot(vector<double> v1, vector<double> v2)
{
    double output = 0.0;
    for(int i = 0; i < size(v1); ++i) { output += v1[i]*v2[i]; }

    return output;
}

// TODO: Add shape checks 
vector<vector<double>> dot(vector<vector<double>> m1, vector<vector<double>> m2)
{   
    vector<vector<double>> output(size(m1));
    for(int i = 0; i < size(m1); ++i)  // each row of m1 and column of m2
    {
        for(int j = 0; j < size(m2[0]); ++j)  // each value in row of m1
        {   
            vector<double> m2_column;
            for(int k = 0; k < size(m2); ++k){ m2_column.push_back(m2[k][j]); }  // creates vertical vector
            output[i].push_back(dot(m1[i], m2_column));  // dot product of m1 row and m2 column
        }
    }
    return output;
}

vector<vector<double>> add(vector<vector<double>> m, vector<double> v)
{
    for(int i = 0; i < size(m); ++i)  // each row of m1
    {
        for(int j = 0; j < size(m[0]); ++j)
        {
            m[i][j] += v[j];
        }
    }
    return m;
}

vector<vector<double>> transpose(vector<vector<double>> m)
{
    vector<vector<double>> output(size(m[0]));
    for(int i = 0; i < size(m); ++i)
    {
        for(int j = 0; j < size(m[0]); ++j)
        {
            output[j].push_back(m[i][j]);
        }
    }
    return output;
}

void pp(vector<vector<double>> m)
{
    for(int i = 0; i < size(m); ++i)
    {
        for(int j = 0; j < size(m[0]); ++j)
        {
            cout << m[i][j] << "   ";
        }
        cout << '\n';
    }
}

void pp(vector<double> v)
{
   for(int i = 0; i < size(v); ++i)
    {
        cout << v[i] << endl;
    }
}

bool test_dot_matrix()
{
    vector<vector<double>> m1 = {
        {0.49, 0.97, 0.53, 0.05},
        {0.33, 0.65, 0.62, 0.51},
        {1.00, 0.38, 0.61, 0.45},
        {0.74, 0.27, 0.64, 0.17},
        {0.36, 0.17, 0.96, 0.12}
    };

    vector<vector<double>> m2 = {
        {0.79, 0.32, 0.68, 0.90, 0.77},
        {0.18, 0.39, 0.12, 0.93, 0.09},
        {0.87, 0.42, 0.60, 0.71, 0.12},
        {0.45, 0.55, 0.40, 0.78, 0.81}
    };

    vector<vector<double>> result = dot(m1, m2);
    return true;
} 

bool test_add()
{
    vector<double> v = {2.0, 3.0, 0.5};
    vector<vector<double>> m = {
        {2.8, -1.8, 1.89, 0}, 
        {6.9, -4.81, -0.3, 0},
        {-0.59, -1.95, -0.47, 0}
    };

    vector<vector<double>> excepted = {
        {4.8, 1.21, 2.385},
        {8.9, -1.81, 0.2},
        {1.41, 1.051, 0.026}
    };

    vector<vector<double>> output = add(m, v);
}

bool test_transpose()
{
    vector<vector<double>> m = {
        {0.79, 0.32, 0.68, 0.90, 0.77},
        {0.18, 0.39, 0.12, 0.93, 0.09},
        {0.87, 0.42, 0.60, 0.71, 0.12},
        {0.45, 0.55, 0.40, 0.78, 0.81}
    };

    vector<vector<double>> expected = {
        {0.79, 0.18, 0.87, 0.45},
        {0.32, 0.39, 0.42, 0.55},
        {0.68, 0.12, 0.60, 0.40},
        {0.90, 0.93, 0.71, 0.78},
        {0.77, 0.09, 0.12, 0.81}
    };

    vector<vector<double>> result = transpose(m);
    return true;
}

class Layer_Dense
{
    private:
        double m_inputs, m_neurons;
        vector<double> m_biases;
        vector<double> generate_biases();
        vector<vector<double>> m_weights;
        vector<vector<double>> generate_weights();

    public:
        Layer_Dense(int& inputs, int& neurons): m_inputs(inputs), m_neurons(neurons)
        {
            this->m_biases = this->generate_biases();
            this->m_weights = this->generate_weights();
        }

        vector<double> getbiases() { return this->m_biases; };
        vector<vector<double>> getweights() { return this->m_weights; };
        vector<vector<double>> forward(vector<vector<double>>& input);
};

vector<vector<double>> Layer_Dense::generate_weights()
{
    srand(0);
    // srand(time(NULL));
    vector<vector<double>> weights(this->m_inputs);
    for(int i = 0; i < this->m_inputs; ++i)
    {
        for(int j = 0; j < this->m_neurons; ++j)
        {
            weights[i].push_back(static_cast<double>(rand() % 100) / 1000);
        }
    } 
    return weights;
}

vector<double> Layer_Dense::generate_biases()
{
    vector<double> biases(this->m_neurons, 0.0);
    return biases;
}

vector<vector<double>> Layer_Dense::forward(vector<vector<double>>& input)
{
    vector<vector<double>> output;
    output = add(dot(input, this->m_weights), this->m_biases);
    return output;
}


int main()
{
    // inputs from input layer or previous neuron layer output
    vector<vector<double>> X = {
        {1.0, 2.0, 3.0, 2.5},
        {2.0, 5.0, -1.0, 2.0},
        {-1.5, 2.7, 3.3, -0.8}
    };

    // bias and weight set for each neuron in the first neuron layer
    vector<double> bias1 = {2.0, 3.0, 0.5};
    vector<vector<double>> weights1 = {
        {0.2, 0.8, -0.5, 1.0},
        {0.5, -0.91, 0.26, -0.5},
        {-0.26, -0.27, 0.17, 0.87}
    };

    // bias and weight set for each neuron in the second neuron layer
    vector<double> bias2 = {-1, 2, -0.5};
    vector<vector<double>> weights2 = {
        {0.1, -0.14, 0.5},
        {-0.5, 0.12, -0.33},
        {-0.44, 0.73, -0.13}
    };

    // vector<vector<double>> layer1 = add(dot(X, transpose(weights1)), bias1);
    // pp(add(dot(layer1, transpose(weights2)), bias2));

    int a = 4;
    int b = 5;
    int c = 2;
    Layer_Dense ld1 = Layer_Dense(a, b);
    Layer_Dense ld2 = Layer_Dense(b, c);

    vector<vector<double>> layer1_output = ld1.forward(X);
    pp(layer1_output);
    cout << '\n';
    pp(ld2.forward(layer1_output));
        
    return 0;

}