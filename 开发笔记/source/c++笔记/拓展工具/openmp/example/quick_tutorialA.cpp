#include <cmath>
#include <iostream>
  int main()
  {
    const int size = 256;
    double sinTable[size];
    
    #pragma omp parallel for
    for(int n=0; n<size; ++n)
      sinTable[n] = std::sin(2 * M_PI * n / size);
    
    #pragma omp parallel
    {
      std::cout << "Hello" << std::endl;
    }
    // the table is now initialized
  }
