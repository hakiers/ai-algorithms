#include "optimizer/GradientDescent.h"
#include "model/LinearRegression.h"
#include <vector>
#include <stdexcept>
#include <algorithm>

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
        for(int batch_start = 0; batch_start < m; batch_start += batch_size){
            int batch_end = std::min(batch_start + batch_size, m);
            int current_batch_size = batch_end - batch_start;

            std::vector<double> error(current_batch_size);
            std::vector<double> newWeights = model.getWeights();
            for(size_t i = batch_start; i < batch_end; i++)
                error[i-batch_start] = f(model.getWeights(), model.getBias(), X[i]) - y[i];

            for(size_t i = 0; i < n; i++){
                double gradient = 0.0;
                for(int j = 0; j < current_batch_size; j++)
                    gradient += error[j] * 2 * X[j][i];
                gradient /= current_batch_size;
                newWeights[i] -= gradient * learning_rate;
            }

            double gradient = 0.0;
            double newBias = model.getBias();
            for(size_t i = 0; i < current_batch_size; i++)
                gradient += error[i] * 2;
            gradient /= current_batch_size;
            newBias -= gradient * learning_rate;

            model.setWeights(newWeights);
            model.setBias(newBias);
        }
    }
}