#include <vector>
#include <iostream>
#include <math.h>
#include "model/LinearRegression.h"
#include "optimizer/GradientDescent.h"
#include "utils/Loss.h"
#include "utils/CSVReader.h"

void standardize(std::vector<std::vector<double>>& X) {
    if (X.empty()) return;

    int m = X.size();
    int n = X[0].size();

    for (int j = 0; j < n; j++) {
        // Compute mean for feature j
        double sum = 0.0;
        for (int i = 0; i < m; i++) {
            sum += X[i][j];
        }
        double mean = sum / m;

        // Compute standard deviation for feature j
        double sq_sum = 0.0;
        for (int i = 0; i < m; i++) {
            sq_sum += (X[i][j] - mean) * (X[i][j] - mean);
        }
        double stddev = std::sqrt(sq_sum / m);

        // Divided by 0
        if (stddev == 0.0) stddev = 1.0;

        // Standarize each feature of value j
        for (int i = 0; i < m; i++) {
            X[i][j] = (X[i][j] - mean) / stddev;
        }
    }
}

int main() {
    std::vector<std::vector<double>> X;
    std::vector<double> Y;

	CSVReader reader("data/air_quality.csv");
	reader.readData(X, Y);

    standardize(X);
    LinearRegression model;
    GradientDescent optimizer(0.01);

    model.fit(X, Y, optimizer, 1000);

    auto predictions = model.predict(X);

    double loss = mseLoss(predictions, Y);
    std::cout << "Final MSE Loss: " << loss << std::endl;

    auto w = model.getWeights();
    std::cout << "Weights: ";
    for (auto weight : w) std::cout << weight << " ";
    std::cout << "\nBias: " << model.getBias() << std::endl;

    std::cout << "Model score: " << model.score(X, Y) << std::endl;
    return 0;
}