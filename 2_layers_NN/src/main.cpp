#include <iostream>
#include <Matrix.h>
#include <Random.hpp>
#include <cmath>
#include <random>
#include <chrono>

#define NB_ELEMENTS 5.0f
#define EPOCH 100
#define LEARNING_RATE 0.01f

void loadDataset(Matrix& X, Matrix& Y, const int nb_elements) {
// Création du dataset de plantes toxiques ou saine
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
			float x1,x2;
			// Génération de données normalisées pour une plante toxique
			if(randomi(0,1)) x1 = randomf(0,20)/60.f;
			else x1 = randomf(40,60)/60.f;
			if(randomi(0,1)) x2 = randomf(0,5)/15.f;
			else x2 = randomf(10,15)/15.f;

			X.setCoeff(0,i,x1); // Longueur feuille
			X.setCoeff(1,i,x2); // Largeur feuille
			
			Y.setCoeff(0,i,1);
		}
		else {
			// Génération de données normalisées pour une plante saine
			X.setCoeff(0,i,randomf(20,40)/60.f); // Longueur feuille
			X.setCoeff(1,i,randomf(5,10)/15.f); // Largeur feuille
			Y.setCoeff(0,i,0);
		}
	}
}

// Réseaux à deux couches
//  
//	w1 ___o
//	   \ / \
//		x	o -> Sortie
//	   / \ /
//	w2 __ o
//

/*
	Dataset: 
	  - 2 classes : toxique et sain
	  - 2 attributs : longueur et largeur feuilles
*/

int main() {
	
	// Phase d'initialisation
	srand(time(NULL));
	Matrix X=Matrix(2, NB_ELEMENTS);
	Matrix Y=Matrix(1, NB_ELEMENTS);
	loadDataset(X, Y, NB_ELEMENTS);

	Matrix W1 = Matrix(2,2, 0.0f);
	Matrix b1 = Matrix(2,X.col(), 1.0f);
	Matrix W2 = Matrix(1,2, 0.0f);
	Matrix b2 = Matrix(1,X.col(), 1.0f);

	for(int i=0; i<W1.row(); i++) {
		for(int j=0; j<W1.col(); j++) {
			W1.setCoeff(i,j, randomf(0,5));
		}	
	}
	
	float val1=randomf(0,5), val2=randomf(0,5);
	for(int i=0; i<b1.row(); i++) {
		for(int j=0; j<b1.col(); j++) {
			if(i) b1.setCoeff(i,j, val1);
			else b1.setCoeff(i,j, val2);
		}
	}

	for(int i=0; i<W2.row(); i++) {
		for(int j=0; j<W2.col(); j++) {
			W2.setCoeff(i,j, randomf(0,5));
		}	
	}

	val1=randomf(0,5);
	for(int j=0; j<b2.col(); j++) {
		b2.setCoeff(0,j, val1);
	}

	W1.disp();
	b1.disp();
	W2.disp();
	b2.disp();

	for(int e=0;e<EPOCH; e++) {
		// Phase forward propagation
			Matrix Z1 = W1*X+b1;
			Matrix A1 = Z1;
			A1.applySigmo();

			Matrix Z2 = W2*A1+b2;
			Matrix A2 = Z2;
			A2.applySigmo();

		// Phase back propagation
			// Seconde couche
			Matrix dZ2 = A2;
			A2 -= Y;

			Matrix dW2 = dZ2*A1.transposee();
			dW2.constMult(1.0f/NB_ELEMENTS);

			Matrix db2=Matrix(1,1,0.0f);
			for(int i=0; i<dZ2.row(); i++) {
				for(int j=0; j<dZ2.col(); j++) {
					db2.setCoeff(0,0, db2.getCoeff(0,0)+dZ2.getCoeff(i,j));
				}
			}
			db2.constMult(1.0f/NB_ELEMENTS);

			// Première couche
			Matrix dZ1=W2.transposee()*dZ2; // *A1(1-A1)
			for(int i=0; i<A1.row(); i++) {
				for(int j=0; j<A1.col(); j++) {
					dZ1.setCoeff(i,j,dZ1.getCoeff(i,j)*A1.getCoeff(i,j)*(1-A1.getCoeff(i,j)));
				}
			}

			Matrix dW1 = dZ1 * X.transposee();
			dW1.constMult(1.0f/NB_ELEMENTS);

			Matrix db1 = Matrix(dZ1.row(),1,0.0f);
			for(int i=0; i<dZ1.row(); i++) {
				for(int j=0; j<dZ1.col(); j++) {
					db1.setCoeff(i,0, db1.getCoeff(i,0)+dZ1.getCoeff(i,j));
				}
			}
			db1.constMult(1.0f/NB_ELEMENTS);

		// Phase d'update du réseaux
			dW1.constMult(LEARNING_RATE); dW2.constMult(LEARNING_RATE);
			W1-=dW1;
			W2-=dW2;
			b2 = Matrix(b2.row(),b2.col(), b2.getCoeff(0,0)-db2.getCoeff(0,0)*LEARNING_RATE);
			float b11 = b1.getCoeff(0,0);
			float b12 = b1.getCoeff(1,0);
			for(int i=0; i<b1.col();i++) {
				b1.setCoeff(0,i,b11-db1.getCoeff(0,0)*LEARNING_RATE);
				b1.setCoeff(1,i,b12-db2.getCoeff(0,0)*LEARNING_RATE);
			}
	}

	// Prédictions
	Matrix X_test=Matrix(2, NB_ELEMENTS);
	Matrix Y_test=Matrix(1, NB_ELEMENTS);
	loadDataset(X_test, Y_test, NB_ELEMENTS);
	
	Matrix Z1 = W1*X_test+b1;
	Matrix A1 = Z1;
	A1.applySigmo();

	Matrix Z2 = W2*A1+b2;
	Matrix A2 = Z2;
	A2.applySigmo();

	A2.disp();

	return 0;
}