#include "optimizer/GradientDescent.h"
#include "model/LinearRegression.h"
#include <vector>
#include <stdexcept>

double f(const std::vector<double> &weights, const double &bias, const std::vector<double> &x){
    double sum = bias;
    for(int i = 0; i < weights.size(); i++)
        sum += weights[i] * x[i];
    return sum;
}

void GradientDescent::setLearningRate(double learning_rate){
    this->learning_rate = learning_rate;
}

void GradientDescent::setBatchSize(int batch_size){
    this->batch_size = batch_size;
}

void GradientDescent::optimize(LinearRegression& model,
                               const std::vector<std::vector<double>>& X,
                               const std::vector<double>& y,
                               int epochs)
{
    int m = X.size();
    int n = X[0].size();

    for(size_t epoch = 0; epoch < epochs; epoch++){
        std::vector<double> error(m);
        std::vector<double> newWeights = model.getWeights();
        for(int i = 0; i < m; i++)
            error[i] = f(model.getWeights(), model.getBias(), X[i]) - y[i];

        for(size_t i = 0; i < n; i++){
            double gradient = 0.0;
            for(int j = 0; j < m; j++)
                gradient += error[j] * 2 * X[j][i];
            gradient /= m;
            newWeights[i] -= gradient * learning_rate;
        }

        double gradient = 0.0;
        double newBias = model.getBias();
        for(size_t i = 0; i < m; i++)
            gradient += error[i] * 2;
        gradient /= m;
        newBias -= gradient * learning_rate;

        model.setWeights(newWeights);
        model.setBias(newBias);
    }
}