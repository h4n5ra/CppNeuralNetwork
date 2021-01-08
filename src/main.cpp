#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include "DenseLayer.h"
#include "Activation.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::RowVector3d;
using Eigen::RowVectorXd;

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

    DenseLayer ld1 = DenseLayer(4, 5);
    DenseLayer ld2 = DenseLayer(5, 2);

    ld1.forward(X);
    MatrixXd layer1_output = ld1.getoutput();
    ld2.forward(layer1_output);
    MatrixXd layer2_output = ld2.getoutput();

    // cout << layer2_output << endl;
    DenseLayer layer1 = DenseLayer(4, 5);
    layer1.forward(X);
    MatrixXd l1out = layer1.getoutput();
    cout << l1out << endl;
    cout << '\n' << endl;
    
    Activation activation1 = Activation();
    cout << activation1.forward_relu(l1out) << endl;
    cout << '\n' << endl;
    
    RowVectorXd smv(3);
    smv << 4.8, 1.21, 2.385;
    cout << activation1.forward_softmax(smv) << endl;
    cout << '\n' << endl;

    MatrixXd smm(3, 3);
    smm << 4.8, 1.21, 2.385,
           8.9, -1.81, 0.2,
           1.41, 1.051, 0.026;
    cout << activation1.forward_softmax(smm) << endl;

    return 0;
}
