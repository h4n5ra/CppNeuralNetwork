#ifndef __ACTIVATION_H_INCLUDED__
#define __ACTIVATION_H_INCLUDED__

#include <string>
#include <Eigen/Dense>
#include "DenseLayer.h"

using Eigen::MatrixXd;
using Eigen::RowVector3d;
using Eigen::RowVectorXd;

class Activation
{

public:
    Activation() {};

    MatrixXd forward_relu(MatrixXd input);
    MatrixXd forward_softmax(MatrixXd input);

    RowVectorXd forward_relu(RowVectorXd input);
    RowVectorXd forward_softmax(RowVectorXd input);
    
};

RowVectorXd Activation::forward_relu(RowVectorXd input)
{
    return input.unaryExpr([](double x){return x>0 ? x : 0.0;});
}

MatrixXd Activation::forward_relu(MatrixXd input)
{
    return input.unaryExpr([](double x){return x>0 ? x : 0.0;});
}

MatrixXd Activation::forward_softmax(MatrixXd input)
{
    double norm, max;
    
    for (int i = 0; i < 3; i++)
    {
        max = input.row(i).maxCoeff();
        input.row(i) = input.row(i) - RowVectorXd::Constant(3, max);
    }

    input = input.unaryExpr([](double x){return std::pow(2.71828, x);});

    for (int i = 0; i < 3; i++)
    {
        norm = input.row(i).sum();
        input.row(i) = input.row(i)/norm;
    }
    
    return input;
}


RowVectorXd Activation::forward_softmax(RowVectorXd input)
{
    input = input - RowVectorXd::Constant(3, input.maxCoeff());
    input = input.unaryExpr([](double x){return std::pow(2.71828, x);});

    return input/input.sum();
}

// TODO:
//   - Remove fixed size for ::Constant
//   - Use high precision exp?
//   - 

#endif
