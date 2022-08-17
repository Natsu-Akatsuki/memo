#include <iostream>
using namespace std;

template <typename T> void write(char *&buffer, const T &val) {
  // T buffer[] = val;
  *reinterpret_cast<T *>(buffer) = val;
  buffer += sizeof(T);
}

void serialize(void *buffer) {
  char *d = static_cast<char *>(buffer);
  char *a = d;
  int mClassCount = 3;
  int mThreadCount = 4;
  write(d, mClassCount);
  write(d, mThreadCount);
}

int main(int argc, char **argv) {

  int num = 0x00636261;
  int *pnum = &num;
  cout << "value of pnum pointer(address): " << pnum << endl;

  // pnum(pointer to int) -> pstr(pointer to char)
  char *pstr = reinterpret_cast<char *>(pnum);
  cout << "value of pstr pointer(address): " << pstr << endl;
  // output pstr directly will output the string it points to
  // use casting to ensure to output the value of pstr(i.e address)
  cout << "value of pstr pointer(address) with static_cast: "
       << static_cast<void *>(pstr) << endl;

  char *p = "empty";
  serialize(&p);
}
