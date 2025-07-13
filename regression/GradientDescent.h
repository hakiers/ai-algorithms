#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H
#include "Optimizer.h"


class GradientDescent : public Optimizer {
    private:
        double learning_rate;
    public:
        GradientDescent(double lr = 0.01) : learning_rate(lr) {}
        void setLearningRate(double);

    void optimize(LinearRegression &model,
                  const std::vector<std::vector<double>> &X,
                  const std::vector<double> &Y,
                  int epochs) override;
};


#endif //GRADIENTDESCENT_H
