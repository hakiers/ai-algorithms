#include "LinearRegression.h"
#include "Optimizer.h"
#include <stdexcept>

long double LinearRegression::predict(const std::vector<long double> &X) const{
    if(!is_trained) throw std::runtime_error("LinearRegression::predict: not trained");
    if(X.size() != weights.size()) throw std::runtime_error("LinearRegression::predict: X.size() != weights.size()");

    long double sum = bias;
    for(int i = 0; i < X.size(); i++)
        sum += X[i] * weights[i];

    return sum;
}

std::vector<long double> LinearRegression::predict(const std::vector<std::vector<long double>> &X) const{
    if(!is_trained) throw std::runtime_error("LinearRegression::predict: not trained");

    std::vector<long double> predictions(X.size());
    for(int i = 0; i < X.size(); i++)
        predictions[i] = predict(X[i]);

    return predictions;
}

void LinearRegression::fit(const std::vector<std::vector<long double>> &X, const std::vector<long double> &Y,
                        Optimizer &optimizer, int epochs)
{
    if(X.size() != Y.size()) throw std::runtime_error("LinearRegression::fit: X.size() != Y.size()");

    weights.resize(X[0].size(), 0.0);
    bias = 0.0;

    optimizer.optimize(*this, X, Y, epochs);

    is_trained = true;
}

std::vector<long double> LinearRegression::getWeights() const{
    return weights;
}

LinearRegression LinearRegression::setWeights(const std::vector<long double> &newWeights){
    weights = newWeights;
    return *this;
}

long double LinearRegression::getBias() const{
    return bias;
}

LinearRegression LinearRegression::setBias(long double bias){
    this->bias = bias;
    return *this;
}

LinearRegression LinearRegression::reset(){
    is_trained = false;
    weights.clear();
    bias = 0.0;
    return *this;
}

