#include "matrix.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <unistd.h>
#include <sys/time.h>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

using namespace std;

QSMatrix<double> fill_matrix(QSMatrix<double> T)
{
	unsigned int N = T.get_cols();
    srand(time(NULL));
    T(0,0) = ::rand() % 200;
    T(0,1) = T(1,0) = rand() % 100;
    for (unsigned int i = 1; i < N - 1; i++){
        T(i,i) = T(i-1,i-1) - rand() % 200;
        T(i+1,i)=T(i,i+1)=rand() % 100;
    }
    T(N-1,N-1)=T(N-2,N-2) - rand() % 200;
    return T;
}

void print(std::string s, QSMatrix<double> M){
	cout.precision(17);
	cout << s << std::endl;

	for (unsigned int i=0; i<M.get_rows(); i++) {
		for (unsigned int j=0; j<M.get_cols(); j++) {
			std::cout << std::fixed << M(i,j) << " ";
		}
		std::cout << std::endl;
	}
	return;
}

std::vector<QSMatrix<double> >  divide(QSMatrix<double> T){
	std::vector <QSMatrix<double> > Parts;
	unsigned int N = T.get_cols();
	int v = T(N/2-1,N/2);
	QSMatrix<double> T1(N/2, N/2, 0);
	QSMatrix<double> T2(N - N/2, N - N/2, 0);
	QSMatrix<double> V(N, 1, 0);
	QSMatrix<double> b(1, 1, 0);
	for (unsigned int i=0; i<N/2; i++) {
		for (unsigned int j=0; j<N/2; j++) {
			T1(i,j) = T(i,j);
		}
	}
	for (unsigned int i=N/2; i<N; i++) {
		for (unsigned int j=N/2; j<N; j++) {
			T2(i - N/2,j - N/2) = T(i,j);
		}
	}
	V(N/2-1,0) = V(N/2, 0) = 1;
	b(0,0) = v;
	T1(N/2-1,N/2-1) -= v;
	T2(0,0) -= v;
	Parts.push_back(T1);
	Parts.push_back(T2);
	Parts.push_back(V);
	Parts.push_back(b);
	return Parts;
}

QSMatrix<double> merge(QSMatrix<double> A_1, QSMatrix<double> A_2){
	unsigned int len_1 = A_1.get_cols();
	unsigned int len_2 = A_2.get_cols();
	QSMatrix<double> D(len_1 + len_2, len_1 + len_2, 0);
	for (unsigned int i = 0; i < len_1; i++) {
		for (unsigned int j = 0; j < len_1; j++){
			D(i,j) = A_1(i,j);
		}
	}
	for (unsigned int i = 0; i < len_2; i++) {
		for (unsigned int j = 0; j < len_2; j++){
			D(len_1 + i,len_1 + j) = A_2(i,j);
		}
	}
	return D;
}



std::vector<double> getFunc(QSMatrix<double> D, QSMatrix<double> U, double b, double x){
	std::vector<double> func(2);
	func[0] = 1.0;
	func[1] = 0.0;
	unsigned len = D.get_cols();
	for (unsigned i = 0; i < len; i++){
		func[0] += b * pow(U(i,0),2) / (D(i,i) - x);
		func[1] += b * pow(U(i,0),2) / pow((D(i,i) - x),2);	
	}
	return func;
}

double getEigenVal(QSMatrix<double> D, QSMatrix<double> U, double b, unsigned i){
	std::vector<double> func(2);
	double eigenVal = D(i,i)+0.00001;
	double errorVal = D(i,i);
	func = getFunc(D,U,b,eigenVal);
	unsigned count = 0;
	while(fabs(func[0]) >= 0.0000001 && count <= 50){
		eigenVal -= func[0] / func[1];
		count++;
		//cout<<eigenVal<<endl;
		//cout<<func[0]<<endl;
		if(fabs(eigenVal) >= 10000){return errorVal;}
		func = getFunc(D,U,b,eigenVal);
	}
	return eigenVal;
}

QSMatrix<double> getInvertMatrix(QSMatrix<double> M, double val){
	unsigned N = M.get_cols();
	QSMatrix<double> L(N, N, 0);
	for (unsigned i = 0; i < N; i++){
		L(i,i) = 1.0/(M(i,i)-val);
	}
	return L;
}

QSMatrix<double> getEigenVectors(QSMatrix<double> D, QSMatrix<double> U, QSMatrix<double> A){
	unsigned N = D.get_cols();
	QSMatrix<double> Q(N, N, 0);
	QSMatrix<double> L(N, 1, 0);
	for (unsigned i = 0; i < N; i++){
		L = getInvertMatrix(D, A(i,i))*U;
		for (unsigned j = 0; j < N; j++){
			Q(j,i) = L(j,0);
		}
	}
	return Q;
}

std::vector<QSMatrix<double> > recursion(QSMatrix<double> T){
	unsigned int N = T.get_cols();
	std::vector <QSMatrix<double> > out;
	std::vector <QSMatrix<double> > out_1;
	std::vector <QSMatrix<double> > out_2;
	std::vector <QSMatrix<double> > parts;
	QSMatrix<double> Q(N, N, 0);
	QSMatrix<double> A(N, N, 0);
	QSMatrix<double> D(N, N, 0);
	QSMatrix<double> U(N, 1, 0);
	if (N == 1){
		Q(0,0) = 1;
		A = T;
	} else {
		parts = divide(T);
		
		omp_set_num_threads(8);
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				out_1 = recursion(parts[0]);
            }
            #pragma omp section
            {
				out_2 = recursion(parts[1]);
			}
		}
		D = merge(out_1[1],out_2[1]);
		U = merge(out_1[0].transpose(),out_2[0].transpose())*parts[2];
		//cout<<"dsfdsf"<<endl;
		for (unsigned i = 0; i < N; i++){
			A(i,i) = getEigenVal(D,U,parts[3](0,0),i);
		}
		//cout<<"1111"<<endl;
		Q = merge(out_1[0],out_2[0])*getEigenVectors(D,U,A);
	}
	//print("Q",Q);
	//print("T",T);
	//print("A",A);
	out.push_back(Q);
	out.push_back(A);
	return out;
}


int main(int argc, char **argv) {
	std::vector <QSMatrix<double> > out;
	struct timeval start, end;
	double delta;

	for(unsigned i = 4; i <= 1024; i*=2){	
		QSMatrix<double> T(i, i, 0);
		T = fill_matrix(T);
		//print("T",T);
		gettimeofday(&start, NULL);
		out = recursion(T);
		gettimeofday(&end, NULL);
		
		delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
		printf("\n size - %d, time - %ff",i,delta);
	}
	//print("Q",out[0]);
	//print("A",out[1]);
  return 0;
}
