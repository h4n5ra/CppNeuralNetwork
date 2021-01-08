#ifndef __DENSELAYER_H_INCLUDED__
#define __DENSELAYER_H_INCLUDED__

#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::RowVector3d;
using Eigen::RowVectorXd;

class DenseLayer
{
    private:
        int m_inputs, m_neurons;
        RowVectorXd m_biases;
        RowVectorXd generate_biases();
        MatrixXd m_weights, m_output;
        MatrixXd generate_weights();

    public:
        DenseLayer(int inputs, int neurons): m_inputs(inputs), m_neurons(neurons)
        {
            this->m_biases = this->generate_biases();
            this->m_weights = this->generate_weights();
        }

        RowVectorXd getbiases() { return this->m_biases; };
        MatrixXd getweights() { return this->m_weights; };
        MatrixXd getoutput() { return this->m_output; };
        void forward(MatrixXd& input);
};

#endif