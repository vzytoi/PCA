#pragma once

#include <vector>
#include <stdexcept>
#include <cmath>

struct Vec {
    std::vector<double> data;

    Vec() = default;

    Vec(int n) {
        data.resize(n);
        for(int i = 0; i < n; i++) {
            data[i]=0;
        }
    }

    double& operator[](int i) { return data[i]; }
    double operator[](int i) const { return data[i]; }

    void resize(int n) { data.resize(n); }

    int size() const { return static_cast<int>(data.size()); }
};

inline Vec operator-(Vec a, const Vec& b) {
    for (int i = 0; i < a.size(); ++i) {
        a[i] = a[i] - b[i];
    }

    return a;
}

inline double dot(Vec u, Vec v) {
    double s = 0.0f;

    for(int i = 0; i < u.size(); i++)
        s += u[i] * v[i];

    return s;
}

// projection de v sur u
Vec proj(Vec v, Vec u) {
    double c = dot(u, v) / dot(u, u);
    Vec p(u.size());

    for(int i = 0; i < u.size(); i++) {
        p[i] = c * u[i];
    }

    return p;
}

// Étant donné v, retourne v / ||v||, ne conserve que l'information
// de l'orientation du vecteur
inline Vec Normalize(Vec v) {
    double norm = std::sqrt(dot(v, v));
    if (norm == 0.0) {
        throw std::runtime_error("Normalize: zero norm");
    }
    for (int i = 0; i < v.size(); ++i) {
        v[i] /= norm;
    }
    return v;
}
