#pragma once

#include <pybind11/pybind11.h>
#include "Vec.h"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>


#define EPSILON 1e-6
#define MAX_STEP_QR 1000
#define N_STEP_II 100

struct Mtx {
    int rows = 0;
    int cols = 0;
    std::vector<Vec> data;

    Mtx() = default;

    Mtx(int r, int c) {
        rows = r;
        cols = c;
        data.resize(c);
        for (int col = 0; col < c; ++col) {
            data[col].resize(r);
        }
    }

    double get(int row, int col) const { return data[col][row]; }
    void set(int row, int col, double v) { data[col][row] = v; }

    Vec& operator[](int col) { return data[col]; }
    const Vec& operator[](int col) const { return data[col]; }

    void print() {
        for(int i = 0; i < this->rows; i++) {
            for(int j = 0; j < this->cols; j++) {
                std::cout << this->get(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    // Retourne la diagonale de la matrice.
    Vec Diag() {
        int n = std::min(this->cols, this->rows);
        int i = 0;
        Vec r(n);

        while (i < n) {
            r[i] = this->get(i, i);
            i++;
        }

        return r;
    }

    // inversion de la matrice:
    // https://fr.wikipedia.org/wiki/Élimination_de_Gauss-Jordan
    Mtx reverse() {
        if (this->rows != this->cols) {
            throw std::runtime_error("Mtx size mismatch");
        }

        int n = this->rows;
        Mtx augmented(n, 2 * n);
        for (int c = 0; c < n; ++c) {
            augmented[c] = this->data[c];
        }

        for (int i = 0; i < n; ++i) {
            augmented.set(i, n + i, 1.0);
        }

        for (int c = 0; c < n; ++c) {
            // find a pivot (partial pivoting)
            int pivot = c;
            double max_abs = std::abs(augmented.get(c, c));
            for (int r = c + 1; r < n; ++r) {
                double v = std::abs(augmented.get(r, c));
                if (v > max_abs) {
                    max_abs = v;
                    pivot = r;
                }
            }
            if (max_abs < 1e-12) {
                throw std::runtime_error("Mtx not inversible");
            }

            // swap rows if necessary
            if (pivot != c) {
                for (int k = 0; k < 2 * n; ++k) {
                    double tmp = augmented.get(pivot, k);
                    augmented.set(pivot, k, augmented.get(c, k));
                    augmented.set(c, k, tmp);
                }
            }

            // normalize pivot row
            double p = augmented.get(c, c);
            for (int k = 0; k < 2 * n; ++k) {
                augmented.set(c, k, augmented.get(c, k) / p);
            }

            // eliminate other rows
            for (int r = 0; r < n; ++r) {
                if (r == c) {
                    continue;
                }
                double factor = augmented.get(r, c);
                for (int k = 0; k < 2 * n; ++k) {
                    augmented.set(r, k, augmented.get(r, k) - factor * augmented.get(c, k));
                }
            }
        }

        Mtx inv(n, n);
        for (int c = 0; c < n; ++c) {
            inv[c] = augmented[n + c];
        }
        return inv;
    }
};

inline Mtx operator*(const Mtx& A, const Mtx& B) {
    if (A.cols != B.rows) {
        throw std::runtime_error("Mtx size mismatch");
    }
    Mtx C(A.rows, B.cols);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < A.cols; ++k) {
                sum += A.get(i, k) * B.get(k, j);
            }
            C.set(i, j, sum);
        }
    }
    return C;
}

inline Vec operator*(const Mtx& A, const Vec& x) {
    if (A.cols != x.size()) {
        throw std::runtime_error("Mtx size mismatch");
    }
    Vec y(A.rows);
    for (int c = 0; c < A.cols; ++c) {
        double xc = x[c];
        for (int r = 0; r < A.rows; ++r) {
            y[r] += A.get(r, c) * xc;
        }
    }
    return y;
}

inline Mtx to_matrix(pybind11::sequence seq2d) {
    Mtx m;
    m.rows = static_cast<int>(pybind11::len(seq2d));
    if (m.rows == 0) {
        return m;
    }

    pybind11::sequence first_row = seq2d[0].cast<pybind11::sequence>();
    m.cols = static_cast<int>(pybind11::len(first_row));
    m.data.resize(m.cols);
    for (int c = 0; c < m.cols; ++c) {
        m.data[c].resize(m.rows);
    }

    for (int i = 0; i < m.rows; ++i) {
        pybind11::sequence row = seq2d[i].cast<pybind11::sequence>();
        int cols = static_cast<int>(pybind11::len(row));
        if (cols != m.cols) {
            throw std::runtime_error("Ragged 2D array: row sizes differ");
        }
        for (int j = 0; j < m.cols; ++j) {
            m.data[j][i] = row[j].cast<double>();
        }
    }

    return m;
}

inline pybind11::list from_matrix(const Mtx& m) {
    pybind11::list out;
    for (int r = 0; r < m.rows; ++r) {
        pybind11::list row;
        for (int c = 0; c < m.cols; ++c) {
            row.append(m.get(r, c));
        }
        out.append(row);
    }
    return out;
}

// Retourne la transposée de A
inline Mtx T(const Mtx& A) {
    Mtx out(A.cols, A.rows);
    for (int r = 0; r < A.rows; ++r) {
        for (int c = 0; c < A.cols; ++c) {
            out.set(c, r, A.get(r, c));
        }
    }
    return out;
}

// On utilise la méthode de Gram-Schmidt
// https://fr.wikipedia.org/wiki/Algorithme_de_Gram-Schmidt
// https://en.wikipedia.org/wiki/Gram–Schmidt_process
Mtx GramSchmidt(Mtx &A) {
    int m = A.rows;
    int n = A.cols;
    Mtx Q(m, n);

    for (int k = 0; k < n; ++k) {
        Vec v = A[k];

        for (int j = 0; j < k; ++j) {
            v = v - proj(v, Q[j]);
        }

        double norm = std::sqrt(dot(v, v));
        if (norm < 1e-12) {
            Vec zero(m);
            Q[k] = zero;
            continue;
        }
        double inv = 1.0 / norm;
        for (int i = 0; i < m; ++i) {
            v[i] *= inv;
        }

        Q[k] = v;
    }

    return Q;
}


// Réalise la décompositon QR de A
// https://fr.wikipedia.org/wiki/Décomposition_QR 
void decomposition_QR(Mtx &A, Mtx &Q, Mtx &R) {
    Q = GramSchmidt(A);
    R = T(Q) * A;
}

inline double calc_error(Mtx &A) {
    double r = 0.0f;
    int n = A.rows < A.cols ? A.rows : A.cols;
    for(int i = 0; i < n - 1; i++) {
        r += A.get(i+1, i) * A.get(i+1, i);
    }

    return r;
}

// prend en entrée une valeur propre et estime le vecteur propre associé
// https://en.wikipedia.org/wiki/Inverse_iteration
Vec inverseIteration(const Mtx &A, double eigen_value) {
    if (A.rows != A.cols) {
        throw std::runtime_error("Mtx size mismatch");
    }
    int n = A.rows;
    // shifted = A - \mu I_n
    Mtx shifted = A;
    for (int i = 0; i < n; ++i) {
        shifted.set(i, i, shifted.get(i, i) - eigen_value);
    }

    // inv = shifted^{-1} = (A - \mu I_n)^{-1}
    Mtx inv = shifted.reverse();

    // v = (1, 0, ..., 0)
    Vec v(n);
    v[0] = 1.0;

    for (int step = 0; step < N_STEP_II; ++step) {
        v = Normalize(inv * v);
    }

    return v;
}

// On commence par approximer les valeurs propres à l'aide de l'algorithme QR
// https://fr.wikipedia.org/wiki/Algorithme_QR
// On s'arrête lorsque la norme de la sous-diagonale est < EPSILON
Vec QR_algorithm(Mtx &M) {
    double error = 1;
    int step = 0;
    Mtx Q, R;
    Mtx A = M;
    
    while (error >= EPSILON && step < MAX_STEP_QR) {
        decomposition_QR(A, Q, R);
        A = R*Q;
        error = calc_error(A);
        step++;
    }

    Vec eigen_values = A.Diag();
    return A.Diag();
}
