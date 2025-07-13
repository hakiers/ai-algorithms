#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <vector>

class Optimizer;

class LinearRegression {
    friend class Optimizer;
    friend class GradientDescent;
    private:
        std::vector<long double> weights;
        long double bias;

        bool is_trained;
    public:
        LinearRegression() : bias(0), is_trained(false) {}
        ~LinearRegression() = default;

        long double predict(const std::vector<long double> &) const;
        std::vector<long double> predict(const std::vector<std::vector<long double>> &) const;

        void fit(const std::vector<std::vector<long double>> &, const std::vector<long double> &,
                 Optimizer &, int);

        std::vector<long double> getWeights() const;
        LinearRegression setWeights(const std::vector<long double> &);

        long double getBias() const;
        LinearRegression setBias(long double);

        LinearRegression reset();
};



#endif //LINEARREGRESSION_H
