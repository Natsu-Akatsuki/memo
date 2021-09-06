#include <iostream>
#include <torch/extension.h>
void fun_hello(void){
  using namespace std;
  cout << "hello world!" << endl;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
  m.def("fun_hello",&fun_hello,"print hello world!");
}
