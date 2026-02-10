#include <pybind11/pybind11.h>
#include "Mtx.h"
#include <iostream>
#include <string>
#include <algorithm>


void test(pybind11::sequence seq) {
    Mtx mtx = to_matrix(seq);
    mtx.print();
    Vec v(mtx.rows);
    for(int i = 0; i < mtx.rows; i++)
        v[i]=0;
}

pybind11::list run(pybind11::sequence seq, int k) {
    Mtx m = to_matrix(seq);

    // calcul de la moyenne
    Vec means(m.cols);

    for(int i = 0; i < m.cols; i++) {
        means[i] = 0;
        for(int j = 0; j < m.rows; j++) {
            means[i] += m.get(j, i);
        }
        means[i] /= double(m.rows);
    }

    // centrage des données
    for(int r = 0; r < m.rows; ++r) {
        for(int c = 0; c < m.cols; ++c) {
            m.set(r, c, m.get(r, c) - means[c]);
        }
    }

    // calul de la matrice de covariance X^T * X
    Mtx cov(m.cols, m.cols);

    for(int j = 0; j < cov.rows; j++) {
        for(int k = 0; k < cov.cols; k++) {
            cov.set(j, k, 0);
            for(int i = 0; i < m.rows; i++) {
                cov.set(j, k, cov.get(j, k)+m.get(i,j)*m.get(i,k));
            }
        }
    }

    Vec eigen_values = QR_algorithm(cov);

    // sort eigenvalues in descending order
    std::sort(eigen_values.data.begin(), eigen_values.data.end(), std::greater<double>());
    Mtx eigen_vectors(m.cols, m.cols);

    // on construit les k vecteurs propres les plus "importants"
    Mtx W(m.cols, k);
    for (int c = 0; c < k; ++c) {
        W[c] = inverseIteration(cov, eigen_values[c]);
    }

    return from_matrix(m * W);
}

pybind11::list inverse(pybind11::sequence seq) {
    Mtx m = to_matrix(seq);
    Mtx inv = m.reverse();
    return from_matrix(inv);
}

PYBIND11_MODULE(pca, m) {
    m.doc() = "Exemple de module C++ pour Python";
    m.def("run", &run, "Renvois les données projetées");
    m.def("inverse", &inverse, "Inverse d'une matrice 2D");
    m.def("test", &test, "Permet de faire des tests");
}
