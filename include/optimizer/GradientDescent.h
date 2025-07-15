#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H
#include "Optimizer.h"


class GradientDescent : public Optimizer {
    private:
        double learning_rate;
        int batch_size;
    public:
        GradientDescent(double lr = 0.01, int bs = 1) : learning_rate(lr), batch_size(bs) {}
        void setLearningRate(double);
        void setBatchSize(int);

    void optimize(LinearRegression &,
                  const std::vector<std::vector<double>> &,
                  const std::vector<double> &,
                  int) override;
};


#endif //GRADIENTDESCENT_H
