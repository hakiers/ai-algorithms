#include "GradientDescent.h"
#include "LinearRegression.h"
#include <vector>
#include <stdexcept>

long double f(const std::vector<long double> &weights, const long double &bias, const std::vector<long double> &x){
    long double sum = bias;
    for(int i = 0; i < weights.size(); i++)
        sum += weights[i] * x[i];
    return sum;
}

void GradientDescent::setLearningRate(long double learning_rate){
    this->learning_rate = learning_rate;
}

void GradientDescent::optimize(LinearRegression& model,
                               const std::vector<std::vector<long double>>& X,
                               const std::vector<long double>& y,
                               int epochs)
{
    int m = X.size();
    int n = X[0].size();

    for(int epoch = 0; epoch < epochs; epoch++){
        std::vector<long double> error(m);

        for(int i = 0; i < m; i++)
            error[i] = f(model.getWeights(), model.getBias(), X[i]) - y[i];

        for(int i = 0; i < n; i++){
            long double gradient = 0.0;
            for(int j = 0; j < m; j++)
                gradient += error[j] * 2 * X[j][i];
            gradient /= m;
            model.weights[i] -= gradient * learning_rate;
        }

        long double gradient = 0.0;
        for(int i = 0; i < m; i++)
            gradient += error[i] * 2;
        gradient /= m;
        model.bias -= gradient * learning_rate;
    }
}