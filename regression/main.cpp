#include <vector>
#include <iostream>
#include "LinearRegression.h"
#include "GradientDescent.h"
#include "../utils/Loss.h"

void generateData(std::vector<std::vector<long double>>& X, std::vector<long double>& Y, int n) {
    std::srand(0);
    X.clear();
    Y.clear();

    for (int i = 0; i < n; i++) {
        long double x1 = (std::rand() % 100) / 10.0;  // 0.0 - 9.9
        long double x2 = (std::rand() % 100) / 10.0;
        long double x3 = (std::rand() % 100) / 10.0;
        long double x4 = (std::rand() % 100) / 10.0;
		long double x5 = (std::rand() % 100) / 10.0;
		long double x6 = (std::rand() % 100) / 10.0;

        long double noise = ((std::rand() % 100) / 100.0) - 0.5;  // -0.5 - 0.5

        long double y = 1 * x1 + 2 * x2 + 3 * x3 + 4 * x4 + 5 * x5 + 6 * x6 + 2137 + noise;

        X.push_back({x1, x2, x3, x4, x5, x6});
        Y.push_back(y);
    }
}

int main() {
    std::vector<std::vector<long double>> X;
    std::vector<long double> Y;

    generateData(X, Y, 1000); // 1000 próbek

    LinearRegression model;
    GradientDescent optimizer(0.001);

    model.fit(X, Y, optimizer, 100000);

    auto predictions = model.predict(X);

    long double loss = mseLoss(predictions, Y);
    std::cout << "Final MSE Loss: " << loss << std::endl;

    auto w = model.getWeights();
    std::cout << "Weights: ";
    for (auto weight : w) std::cout << weight << " ";
    std::cout << "\nBias: " << model.getBias() << std::endl;

    return 0;
}