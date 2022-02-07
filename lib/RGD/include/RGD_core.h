#ifndef INDICATORS_CORE_H
#define INDICATORS_CORE_H

#include <Eigen/Dense>

using Eigen::MatrixXd;
using namespace Eigen;

namespace acls {
    VectorXd GD(VectorXd beta_0,const double tau, MatrixXd X, MatrixXd Y,double eta_0=1e-3,const double alpha=2);
    MatrixXd runif_in_pball(const int n,const int d, const int p, const double r=1);
    VectorXd RGD(MatrixXd X,MatrixXd Y,const double tau,const double iter,double eta_0=1e-3,const double alpha=2);
}

#endif