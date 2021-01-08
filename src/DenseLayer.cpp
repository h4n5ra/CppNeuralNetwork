#include "DenseLayer.h"
#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::RowVector3d;
using Eigen::RowVectorXd;


MatrixXd DenseLayer::generate_weights()
{
    MatrixXd weights;
    weights.setRandom(this->m_inputs, this->m_neurons);
    return weights;
}

RowVectorXd DenseLayer::generate_biases()
{
    RowVectorXd biases;
    biases.setZero(this->m_neurons);
    return biases;
}

void DenseLayer::forward(MatrixXd& input)
{
    MatrixXd output;
    output = input*(this->m_weights);
    for(int i = 0; i < output.rows(); ++i)
    {
        output.row(i) += this->m_biases;
    }
    this->m_output = output;
}
