#include <iostream>
#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::RowVector3d;
using Eigen::RowVectorXd;

class Activation_ReLU
{
private:
    RowVectorXd m_output_vector;
    MatrixXd m_output_matrix;
public:
    Activation_ReLU() {};
    MatrixXd getoutput() { return m_output_matrix; };
    void forward(RowVectorXd& input);
    void forward(MatrixXd& input);
};

void Activation_ReLU::forward(RowVectorXd& input)
{
    this->m_output_vector = input.unaryExpr([](double x){return x>0 ? x : 0.0;});
}

void Activation_ReLU::forward(MatrixXd& input)
{
    this->m_output_matrix = input.unaryExpr([](double x){return x>0 ? x : 0.0;});
}

class Layer_Dense
{
    private:
        int m_inputs, m_neurons;
        RowVectorXd m_biases;
        RowVectorXd generate_biases();
        MatrixXd m_weights, m_output;
        MatrixXd generate_weights();

    public:
        Layer_Dense(int& inputs, int& neurons): m_inputs(inputs), m_neurons(neurons)
        {
            this->m_biases = this->generate_biases();
            this->m_weights = this->generate_weights();
        }

        RowVectorXd getbiases() { return this->m_biases; };
        MatrixXd getweights() { return this->m_weights; };
        MatrixXd getoutput() { return this->m_output; };
        void forward(MatrixXd& input);
};

MatrixXd Layer_Dense::generate_weights()
{
    MatrixXd weights;
    weights.setRandom(this->m_inputs, this->m_neurons);
    return weights;
}

RowVectorXd Layer_Dense::generate_biases()
{
    RowVectorXd biases;
    biases.setZero(this->m_neurons);
    return biases;
}

void Layer_Dense::forward(MatrixXd& input)
{
    MatrixXd output;
    output = input*(this->m_weights);
    for(int i = 0; i < output.rows(); ++i)
    {
        output.row(i) += this->m_biases;
    }
    this->m_output = output;
}

int main()
{
    // inputs from input layer or previous neuron layer output
    MatrixXd X(3,4);
    X << 1.0, 2.0, 3.0, 2.5,
        2.0, 5.0, -1.0, 2.0,
        -1.5, 2.7, 3.3, -0.8;

    // bias and weight set for each neuron in the first neuron layer
    RowVector3d bias1;
    bias1 << 2.0, 3.0, 0.5;
    MatrixXd weights1(3,4);
    weights1 << 0.2, 0.8, -0.5, 1.0,
                0.5, -0.91, 0.26, -0.5,
                -0.26, -0.27, 0.17, 0.87;

    // bias and weight set for each neuron in the second neuron layer
    RowVector3d bias2;
    bias2 << -1, 2, -0.5;
    MatrixXd weights2(3,3);
    weights2 << 0.1, -0.14, 0.5,
                -0.5, 0.12, -0.33,
                -0.44, 0.73, -0.13;

    MatrixXd Layer1 = X*weights1.transpose();
    for(int i = 0; i < 3; ++i)
    {
        Layer1.row(i) += bias1;
    }

    MatrixXd Layer2 = Layer1*weights2.transpose();
    for(int i = 0; i < 3; ++i)
    {
        Layer2.row(i) += bias2;
    }
    // cout << Layer2 << endl;

    int a = 4;
    int b = 5;
    int c = 2;
    Layer_Dense ld1 = Layer_Dense(a, b);
    Layer_Dense ld2 = Layer_Dense(b, c);

    ld1.forward(X);
    MatrixXd layer1_output = ld1.getoutput();
    ld2.forward(layer1_output);
    MatrixXd layer2_output = ld2.getoutput();

    // cout << layer2_output << endl;
    int d = 4;
    int e = 5;
    Layer_Dense layer1 = Layer_Dense(d, e);
    layer1.forward(X);
    MatrixXd l1out = layer1.getoutput();
    cout << l1out << endl;

    Activation_ReLU activation1 = Activation_ReLU();
    activation1.forward(l1out);
    cout << activation1.getoutput() << endl;


    return 0;

}