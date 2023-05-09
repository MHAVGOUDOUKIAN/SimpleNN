#include <iostream>
#include <Matrix.h>
#include <Random.hpp>
#include <cmath>
#include <random>
#include <chrono>

#define NB_ELEMENTS 10.f
#define EPOCH 10000

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
			// Génration de données normalisées pour une plante toxique
			X.setCoeff(i,0,randomf(40,65)/65.f); // Longueur feuille
			X.setCoeff(i,1,randomf(2,8)/15.f); // Largeur feuille
			Y.setCoeff(i,0,1);
		}
		else {
			// Génration de données normalisées pour une plante saine
			X.setCoeff(i,0,randomf(25,43)/65.f); // Longueur feuille
			X.setCoeff(i,1,randomf(7,15)/15.f); // Largeur feuille
			Y.setCoeff(i,0,0);
		}
	}
}

int main() {
	
	Matrix X=Matrix(NB_ELEMENTS,2);
	Matrix Y=Matrix(NB_ELEMENTS,1);
	loadDataset(X, Y, NB_ELEMENTS);

	Matrix W=Matrix(X.col(), 1);
	for(int i=0; i<X.col();i++) {
		W.setCoeff(i,0, randomf(0,1));
	}

	Matrix b = Matrix(NB_ELEMENTS,1);
	float value = randomf(0,1);
	for(int i=0; i<NB_ELEMENTS; i++) {
		b.setCoeff(i,0,value);
	}

	for(int i=0; i<EPOCH;i++) {
		// Estimation du modèle
		Matrix Z = X*W+b;

		// Calcul de la fonction d'activation
		Matrix A = Z;

		A.applySigmo();

		// Calcul du loss
		double loss;
		for(int i=0; i<NB_ELEMENTS; i++) {
			loss+=Y.getCoeff(i,0)*log(A.getCoeff(i,0))+(1-Y.getCoeff(i,0))*log(1-A.getCoeff(i,0));
		}
		loss *= (-1/NB_ELEMENTS);
		std::cout << "Loss: " << loss << std::endl; 

		// Calcul des gradients
		Matrix tempMat = A;
		tempMat -= Y;
		Matrix dW = X.transposee();
		dW=dW*tempMat;
		dW.constMult(1/NB_ELEMENTS);
		

		float db =0;
		for(int i=0; i<Y.row(); i++) {
			db += A.getCoeff(i,0) - Y.getCoeff(i,0);
		}
		db *= 1/NB_ELEMENTS;

		
		// Mise a jour des paramètres du neurone
		W -= dW;
		for(int i=0; i<b.row(); i++) b.setCoeff(i,0, b.getCoeff(i,0)-db);
	}
	
	// Estimation d'une données
	Matrix X_test=Matrix(NB_ELEMENTS,2);
	Matrix Y_test=Matrix(NB_ELEMENTS,1);
	loadDataset(X_test, Y_test, NB_ELEMENTS);

	Matrix Z = X*W+b;

	// Calcul de la fonction d'activation
	Matrix A = Z;

	A.applySigmo();

	for(int i=0; i<A.row();i++) {
		std::cout << X.getCoeff(i,0)*65.f << "," << X.getCoeff(i,1)*15.f << " Class: "<< Y.getCoeff(i,0) << " Estimation:";
		if(A.getCoeff(i,0)>0.5) std::cout << "Plante Toxique(" << A.getCoeff(i,0) << ")" << std::endl;
		else std::cout << "Plante Saine(" << A.getCoeff(i,0) << ")" << std::endl;
	}

	return 0;
}