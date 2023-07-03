#include "kaldi_align.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
namespace py = pybind11;

static py::dict EditDistance(const std::vector<int> &a,
                             const std::vector<int> &b,
                             const bool sclite_mode) {
  int ins;
  int del;
  int sub;

  int total = LevenshteinEditDistance(a, b, sclite_mode, &ins, &del, &sub);
  py::dict ans;
  ans["ins"] = ins;
  ans["del"] = del;
  ans["sub"] = sub;
  ans["total"] = total;
  return ans;
}

static std::vector<std::pair<int, int>>
Align(const std::vector<int> &a, const std::vector<int> &b, int eps_symbol, const bool sclite_mode) {
  std::vector<std::pair<int, int>> ans;
  LevenshteinAlignment(a, b, eps_symbol, sclite_mode, &ans);
  return ans;
}

static int StreamingEditDistance(const std::vector<std::tuple<int, float, float>> &a,
                                const std::vector<std::tuple<int, float, float>> &b,
                                const float threshold, 
                                const int ins_cost, 
                                const int del_cost, 
                                const int sub_cost, 
                                const int str_cost) {
  return LevenshteinEditDistance(a, b, threshold, ins_cost, del_cost, sub_cost, str_cost);
}

PYBIND11_MODULE(_kaldialign, m) {
  m.doc() = "Python wrapper for kaldialign";
  m.def("edit_distance", &EditDistance, py::arg("a"), py::arg("b"), py::arg("sclite_mode") = false);
  m.def("align", &Align, py::arg("a"), py::arg("b"), py::arg("eps_symbol"), py::arg("sclite_mode") = false);
  m.def("streaming_edit_distance", &StreamingEditDistance, py::arg("a"), py::arg("b"), py::arg("threshold") = 0.0, py::arg("ins_cost") = 1, py::arg("del_cost") = 1, py::arg("sub_cost") = 2, py::arg("str_cost") = 1);
}
