#include <iostream>
#include <Matrix.h>
#include <Random.hpp>
#include <cmath>

#define NB_ELEMENTS 5
#define EPOCH 500

void loadDataset(Matrix& X, Matrix& Y, const int nb_elements) {
// Création du dataset pour déterminer la nature d'une plantes
// en fonction de la longueur et de la largeur de leurs feuilles

	// X : attributs
	// x1 longueur feuille
	// x2 largeur feuille

	// Y : classes
	// classe 0: plante saine
	// classe 1: plante toxique

	for(int i=0; i<nb_elements; i++) {
		if(randomi(0,1))
		{
			X.setCoeff(i,0,randomi(40,65));
			X.setCoeff(i,1,randomi(2,8));
			Y.setCoeff(i,0,0);
		}
		else {
			X.setCoeff(i,0,randomi(25,43));
			X.setCoeff(i,1,randomi(7,15));
			Y.setCoeff(i,0,1);
		}
	}
}

int main() {
	
	Matrix X=Matrix(NB_ELEMENTS,2);
	Matrix Y=Matrix(NB_ELEMENTS,1);
	loadDataset(X, Y, NB_ELEMENTS);

	X.disp();
	Y.disp();

	Matrix W=Matrix(X.col(), 1);
	for(int i=0; i<X.col();i++) {
		W.setCoeff(i,0,randomf(1,5));
	}

	Matrix b = Matrix(NB_ELEMENTS,1);
	for(int i=0; i<NB_ELEMENTS; i++) {
		b.setCoeff(i,0,randomf(1,5));
	}

	b.disp();
	W.disp();

	Matrix Z = X*W+b;

	Z.disp();

	Matrix A = Z;
	A.applySigmo();

	A.disp();

	// Calcul du loss
	float loss;
	for(int i=0; i<NB_ELEMENTS; i++) {
		loss+=Y.getCoeff(i,0)*log(A.getCoeff(i,0))+(1-Y.getCoeff(i,0))*log(1-A.getCoeff(i,0));
	}
	loss *= (-1/NB_ELEMENTS);

	Matrix dW = A;
	dW-=Y;
	dW = dW*X.transposee();
	dW.constMult(1/NB_ELEMENTS);


	return 0;
}