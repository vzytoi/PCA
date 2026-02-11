#include <pybind11/pybind11.h>
#include "Mtx.h"
#include <iostream>
#include <string>
#include <algorithm>

struct PCAResult {
    Mtx projected;
    Mtx W;
    Vec means;
};

PCAResult compute_pca(Mtx m, int k) {
    if (k <= 0 || k > m.cols) {
        throw std::runtime_error("k must be in [1, n_features]");
    }

    PCAResult result;

    // calcul de la moyenne
    Vec means(m.cols);
    for (int i = 0; i < m.cols; i++) {
        means[i] = 0;
        for (int j = 0; j < m.rows; j++) {
            means[i] += m.get(j, i);
        }
        means[i] /= double(m.rows);
    }

    // centrage des données
    for (int r = 0; r < m.rows; ++r) {
        for (int c = 0; c < m.cols; ++c) {
            m.set(r, c, m.get(r, c) - means[c]);
        }
    }

    // calul de la matrice de covariance X^T * X
    Mtx cov(m.cols, m.cols);
    for (int j = 0; j < cov.rows; j++) {
        for (int col = 0; col < cov.cols; col++) {
            cov.set(j, col, 0);
            for (int i = 0; i < m.rows; i++) {
                cov.set(j, col, cov.get(j, col) + m.get(i, j) * m.get(i, col));
            }
        }
    }

    Vec eigen_values = QR_algorithm(cov);

    // sort eigenvalues in descending order
    std::sort(eigen_values.data.begin(), eigen_values.data.end(), std::greater<double>());

    // on construit les k vecteurs propres les plus "importants"
    Mtx W(m.cols, k);
    for (int c = 0; c < k; ++c) {
        W[c] = inverseIteration(cov, eigen_values[c]);
    }

    result.projected = m * W;
    result.W = W;
    result.means = means;
    return result;
}

pybind11::list from_vec(const Vec& v) {
    pybind11::list out;
    for (int i = 0; i < v.size(); ++i) {
        out.append(v[i]);
    }
    return out;
}

pybind11::dict fit(pybind11::sequence seq, int k) {
    Mtx m = to_matrix(seq);
    PCAResult result = compute_pca(m, k);

    pybind11::dict out;
    out["projected"] = from_matrix(result.projected);
    out["W"] = from_matrix(result.W);
    out["means"] = from_vec(result.means);
    return out;
}

PYBIND11_MODULE(pca, m) {
    m.doc() = "Exemple de module C++ pour Python";
    m.def("fit", &fit, "Renvois les données projetees, la base W et les moyennes");
}
