#include <iostream>
#include <Eigen/Dense>
#include <cmath>
 
using Eigen::MatrixXd;
using namespace Eigen;


namespace acls{
VectorXd GD(VectorXd beta_0,const double tau, MatrixXd X, MatrixXd Y,double eta_0=1e-3,const double alpha=2){
    int n=X.rows();
  VectorXd e_0=Y-X*beta_0;
  VectorXd a=e_0.cwiseAbs();
  VectorXd b=a;
  for (int i=0; i<a.size();++i){
  if (a(i)>tau){
      b(i)=tau;
  }
  }
  double L_0=0.5*b.array().square().sum();
  Eigen::Vector<bool,Dynamic> w=a.array()<=tau;
  VectorXd ew=w.cast<double>().array()*e_0.array();
  VectorXd diff=-X.transpose()*ew/n;
  VectorXd beta=beta_0-eta_0*diff;
  VectorXd e_1=Y-X*beta;

  a=e_1.cwiseAbs();
  b=a;
  for (int i=0; i<a.size();++i){
  if (a(i)>tau){
      b(i)=tau;
  }
  }
  double L_1=0.5*b.array().square().sum();
  double eta=alpha*eta_0;
  beta=beta_0-eta*diff;
  VectorXd e_2=Y-X*beta;
  a=e_2.cwiseAbs();
  b=a;
  for (int i=0; i<a.size();++i){
   if (a(i)>tau){
       b(i)=tau;
   }
  }
  double L_2=0.5*b.array().square().sum();

    while (L_2<L_1){
    L_1=L_2;
    eta=alpha*eta;
    beta=beta_0-eta*diff;
    e_2=Y-X*beta;
    a=e_2.cwiseAbs();
    b=a;
    for (int i=0; i<a.size();++i){
        if (a(i)>tau){
          b(i)=tau;
        }
    }
    L_2=0.5*b.array().square().sum();
  }
eta=eta/alpha;
  beta=beta_0-eta*diff;
  if (L_0<L_1){
    beta=beta_0;
  }

  int j=0;
  double d= sqrt((beta_0-beta).array().square().sum()/n);
  while (d > 1e-8  &&  j<10000000){
    beta_0=beta;
    e_0=Y-X*beta_0;
    a=e_0.cwiseAbs();
   for (int i=0; i<a.size();++i){
    if (a(i)>tau){
       b(i)=tau;
    }
  }
    L_0=0.5*b.array().square().sum();
    w=a.array()<=tau;
    ew=w.cast<double>().array()*e_0.array();
    diff=-X.transpose()*ew/n;
    beta=beta_0-eta_0*diff;
    e_1=Y-X*beta;

  a=e_1.cwiseAbs();
  b=a;
  for (int i=0; i<a.size();++i){
  if (a(i)>tau){
      b(i)=tau;
  }
  }
  L_1=0.5*b.array().square().sum();
  eta=alpha*eta_0;
  beta=beta_0-eta*diff;
  e_2=Y-X*beta;
  a=e_2.cwiseAbs();
  b=a;
  for (int i=0; i<a.size();++i){
   if (a(i)>tau){
       b(i)=tau;
   }
  }
  L_2=0.5*b.array().square().sum();

    while (L_2<L_1){
    L_1=L_2;
    eta=alpha*eta;
    beta=beta_0-eta*diff;
    e_2=Y-X*beta;
    a=e_2.cwiseAbs();
    b=a;
    for (int i=0; i<a.size();++i){
        if (a(i)>tau){
          b(i)=tau;
        }
    }
    L_2=0.5*b.array().square().sum();
  }
eta=eta/alpha;
  beta=beta_0-eta*diff;
  if (L_0<L_1){
    beta=beta_0;
  }

    d= sqrt((beta_0-beta).array().square().sum()/n);
    j++;
  }
  return beta;
}

MatrixXd runif_in_pball(const int n,const int d, const int p, const double r=1){
 MatrixXd out;
  out.setOnes(n,d);
  const int m=n*d;
  double l=floor(1.0/p);
  double p1=1.0/p-l;
  double p2=1-p1;
  int i=0;
  VectorXd R;
  R.setZero(m);
  std::srand((unsigned int) time(0));
  for (;i<m;++i){
      double s=0;
      if(l==0){
          s=0;
      } else {
       VectorXd ss=(VectorXd::Random(l).array()+1.)*0.5;
       s=ss.array().log().sum();
      }
     if (l!=1.0/p){
       VectorXd U=(VectorXd::Random(2).array()+1.)*0.5; 
       double a=pow(U(0),(1.0/p1))+pow(U(1),(1.0/p2));
       while (a>1){
        U=(VectorXd::Random(2).array()+1.)*0.5; 
       }
       VectorXd A=(VectorXd::Random(1).array()+1.)*0.5; 
       s=s+(A.array().log().sum())*pow(U(0),1.0/p1)/(pow(U(0),(1.0/p1))+pow(U(1),(1.0/p2)));
     }
     R(i)=pow((-p*s),1.0/p);
  }
  VectorXd A=(VectorXd::Random(m)+VectorXd::Ones(m))*0.5;
  Vector<bool,Dynamic> Ai=(A.array()>0.5);
  A=Ai.cast<double>();
  VectorXd B=A.array()-1;
  VectorXd C=A+B;
  MatrixXd epsilon=R.array()*C.array();
  epsilon.resize(n,d);
  A=(VectorXd::Random(m)+VectorXd::Ones(m))*0.5;
  Ai=(A.array()>0.5);
  A=Ai.cast<double>();
  MatrixXd signs=2*(A.array()+1)-1;
  signs.resize(n,d);
  MatrixXd x=signs.array()*epsilon.array();
  VectorXd zs=(VectorXd::Random(n).array()+1.)*0.5;
  VectorXd z=pow(zs.array(),1.0/d);
   for (int j=0;j<n;++j){
     out.row(j)=r*z(j)*x.row(j)/pow(pow(x.row(j).array(),p).cwiseAbs().sum(),(1.0/p));
   }

return out;
    
 }

  VectorXd RGD(MatrixXd X,MatrixXd Y,const double tau,const double iter,double eta_0=1e-3,const double alpha=2){
  VectorXd beta_tmp=(X.transpose()*X).inverse()*X.transpose()*Y;
  VectorXd beta_0=beta_tmp;
  VectorXd beta=beta_tmp;
  int p=X.cols()-1;
  double L=1000000000;
  beta_0=runif_in_pball(1,p+1,2,tau).transpose();
  for (int s=0; s<iter;s++){
    beta_0=runif_in_pball(1,p+1,2,tau).transpose();
    beta_tmp=GD(beta_0,tau,X,Y,eta_0,alpha);
    VectorXd r=Y-X*beta_tmp;
    VectorXd a=r.cwiseAbs();
    VectorXd b=a;
    for (int i=0; i<a.size();++i){
        if (a(i)>tau){
          b(i)=tau;
        }
    }
    double L_tmp=0.5*b.array().square().sum();
    if (L_tmp<L){
      L=L_tmp;
      beta=beta_tmp;
    }
 }
 return beta;
  }
}
