#include <iostream>
#include "tuple"
#include "vector"
#include <random>
#include <math.h>

typedef std::vector<std::vector<double>> matrix;
typedef std::tuple<std::vector<double>, std::vector<double>> hyper_parameters;
typedef std::tuple<hyper_parameters, std::vector<double>> neural_network_outputs;

hyper_parameters initialize(const matrix& X)
{
    auto nb_samples{ X.size() };
    auto nb_features{ X[0].size() };
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(0.0, 1.0);

    //Initialization of weights vector
    std::vector<double> W(nb_features);
    for (auto i = 0; i < nb_features; ++i) {
        W[i] = dis(gen);
    }

    //Initialization of biais "vector" (in fact a vector based on unique scalar value)
    auto randomNum{ dis(gen) };
    std::vector<double> b(nb_samples, randomNum);

    return std::make_tuple(W, b);
}

std::vector<double> model(const matrix& X, const hyper_parameters& W_b)
{
    auto nb_samples{ X.size() };
    auto nb_features{ X[0].size() };
    std::vector<double> z(nb_samples);
    std::vector<double> a(nb_samples);

    //Z = WX + b
    for (auto i = 0; i < nb_samples; ++i)
    {
        z[i] = 0;
        for (auto j = 0; j < nb_features; ++j)
        {
            z[i] += std::get<0>(W_b)[j] * X[i][j] + std::get<1>(W_b)[i];
        }
    }

    //A = 1 / (1 + exp(-Z))
    //This sigmoid activation function leads to values between [0,1], like probabilities, 
    //and with smooth behavior (contraty to for example Heavyside one)
    for (auto i = 0; i < nb_samples; ++i)
    {
        a[i] = 1.0 / (1 + exp(-z[i]));
    }

    return a;
}

double cross_entropy_cost_function(const std::vector<double>& a, const std::vector<double>& y)
{
    auto l{ 0.0 };
    auto nb_samples{ a.size() };

    //l = -1/m * sum_{i = [1,m]}(y_i * log(a_i) + (1 - y_i) * log(1 - a_i))
    for (auto i = 0; i < nb_samples; ++i)
    {
        l += y[i] * log(a[i]) + (1 - y[i]) * log(1 - a[i]);
    }

    return -1.0 / nb_samples * l;
}

hyper_parameters gradients(const std::vector<double>& a, const std::vector<double>& y, const matrix& X)
{
    auto nb_samples{ X.size() };
    auto nb_features{ X[0].size() };
    std::vector<double> dW(nb_features, 0.0);
    auto db_scalar{ 0.0 };

    //dW(j) = 1/m * sum_{i = [1,m]}(a_i - y_i)*X[j]
    //db = 1/m * sum_{i = [1,m]}(a_i - y_i)
    for (auto j = 0; j < nb_features; ++j)
    {
        for (auto i = 0; i < nb_samples; ++i)
        {
            dW[j] += (a[i] - y[i]) * X[i][j];
            db_scalar += (a[i] - y[i]);
        }
        dW[j] /= nb_samples;
    }

    std::vector<double> db(nb_samples, 1.0 / nb_samples * db_scalar);

    return std::make_tuple(dW, db);
}

void update(const hyper_parameters& dW_db, hyper_parameters& W_b, double alpha)
{
    auto nb_samples{ std::get<1>(W_b).size() };
    auto nb_features{ std::get<0>(W_b).size() };

    //W = W - alpha * dW
    for (auto j = 0; j < nb_features; ++j)
    {
        std::get<0>(W_b)[j] = std::get<0>(W_b)[j] - alpha * std::get<0>(dW_db)[j];
    }

    //b = b - alpha * db
    for (auto i = 0; i < nb_samples; ++i)
    {
        std::get<1>(W_b)[i] = std::get<1>(W_b)[i] - alpha * std::get<1>(dW_db)[i];
    }
}

std::vector<double> predict(const matrix& X, const hyper_parameters& W_b)
{
    auto a{ model(X, W_b) };
    auto nb_samples{ X.size() };
    std::vector<double> y_pred(nb_samples, 0.0);
    //As A embedds probabilities (thenks to sigmoid), prediction values are equal to 0 or 1, according to A threshhold
    for (auto i = 0; i < nb_samples; ++i)
    {
        if (a[i] >= 0.5)
        {
            y_pred[i] = 1.0;
        }
    }
    return y_pred;
}

//This additional function aims to mimic the python sklearn accuracy_score metric (an average of errors summation)
double accuracy_score(const std::vector<double>& y, const std::vector<double>& y_pred)
{
    auto nb_samples{ y.size() };
    auto res{ 0.0 };
    for (auto i = 0; i < nb_samples; ++i)
    {
        if (y[i] == y_pred[i])
        {
            res += 1;
        }
    }
    return res / nb_samples * 100;
}

neural_network_outputs artificial_neuron_perceptron_for_loop(const matrix& X, const std::vector<double>& y, double alpha, int n_iter)
{
    std::vector<double> l(n_iter);
    auto W_b{ initialize(X) };

    for (auto k = 0; k != n_iter; ++k)
    {
        auto a{ model(X, W_b) };
        l[k] = cross_entropy_cost_function(a, y);
        auto dW_db{ gradients(a, y, X) };
        update(dW_db, W_b, alpha);
    }

    auto y_pred{ predict(X, W_b) };
    std::cout << "The accuracy score is " << accuracy_score(y, y_pred) << "%" << std::endl;

    return std::make_tuple(W_b, l);
}

//TO DO : artificial_neuron_perceptron_while_loop, in order to embedd dynamic error check during the loop

int main()
{
    //Use case #1 with linear separability
    matrix x(8);
    x[0] = { 1.0,1.0 };
    x[1] = { 1.1,1.2 };
    x[2] = { 1.2,1.3 };
    x[3] = { 1.4,1.4 };
    x[4] = { 10.0,10.0 };
    x[5] = { 10.1,10.2 };
    x[6] = { 10.2,10.3 };
    x[7] = { 10.4,10.4 };

    std::vector<double> y{ 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0 };

    auto out{ artificial_neuron_perceptron_for_loop(x, y, 0.1, 100) };
    std::cout << "Linear separability error tracking" << std::endl;
    for (auto k = 0; k != 100; ++k)
    {
        std::cout << std::get<1>(out)[k] << std::endl;
    }

    matrix x_to_predict(2);
    x_to_predict[0] = { 1.0,2.0 };
    x_to_predict[1] = { 9.0,9.0 };
    auto sample_pred{ predict(x_to_predict, std::get<0>(out)) };
    std::cout << "Prediction for the first added sample is " << sample_pred[0] << std::endl;
    std::cout << "Prediction for the second added sample is " << sample_pred[1] << std::endl;

    //Use case #2 without linear separability
    matrix x_2(8);
    x_2[0] = { 1.0,1.0 };
    x_2[1] = { 1.1,1.2 };
    x_2[2] = { 1.2,1.3 };
    x_2[3] = { 1.4,1.4 };
    x_2[4] = { 10.0,10.0 };
    x_2[5] = { 10.1,10.2 };
    x_2[6] = { 10.2,10.3 };
    x_2[7] = { 10.4,10.4 };

    std::vector<double> y_2{ 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0 };

    auto out_2{ artificial_neuron_perceptron_for_loop(x_2, y_2, 0.1, 100) };
    std::cout << "No linear separability error tracking" << std::endl;
    for (auto k = 0; k != 100; ++k)
    {
        std::cout << std::get<1>(out_2)[k] << std::endl;
    }

    matrix x_to_predict_2(2);
    x_to_predict_2[0] = { 1.0,2.0 };
    x_to_predict_2[1] = { 9.0,9.0 };
    auto sample_pred_2{ predict(x_to_predict_2, std::get<0>(out_2)) };
    std::cout << "Prediction for the first added sample is " << sample_pred_2[0] << std::endl;
    std::cout << "Prediction for the second added sample is " << sample_pred_2[1] << std::endl;

    //Use case #3 with almost linear separability
    matrix x_3(8);
    x_3[0] = { 1.0,1.0 };
    x_3[1] = { 1.1,1.2 };
    x_3[2] = { 1.2,1.3 };
    x_3[3] = { 1.4,1.4 };
    x_3[4] = { 10.0,10.0 };
    x_3[5] = { 10.1,10.2 };
    x_3[6] = { 10.2,10.3 };
    x_3[7] = { 10.4,10.4 };

    std::vector<double> y_3{ 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0 };

    auto out_3{ artificial_neuron_perceptron_for_loop(x_3, y_3, 0.1, 100) };
    std::cout << "Linear separability error tracking" << std::endl;
    for (auto k = 0; k != 100; ++k)
    {
        std::cout << std::get<1>(out_3)[k] << std::endl;
    }

    matrix x_to_predict_3(2);
    x_to_predict_3[0] = { 1.0,2.0 };
    x_to_predict_3[1] = { 9.0,9.0 };
    auto sample_pred_3{ predict(x_to_predict_3, std::get<0>(out_3)) };
    std::cout << "Prediction for the first added sample is " << sample_pred_3[0] << std::endl;
    std::cout << "Prediction for the second added sample is " << sample_pred_3[1] << std::endl;
}
