#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H
#include "Optimizer.h"


class GradientDescent : public Optimizer {
    private:
        long double learning_rate;
    public:
        GradientDescent(long double lr = 0.01) : learning_rate(lr) {}
        void setLearningRate(long double);

    void optimize(LinearRegression &model,
                  const std::vector<std::vector<long double>> &X,
                  const std::vector<long double> &Y,
                  int epochs) override;
};


#endif //GRADIENTDESCENT_H
