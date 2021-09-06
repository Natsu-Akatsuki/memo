/*
 * 任务：
 * 1.重载加法操作符，输出流<<操作符，使其支持对自定义类型操作数的操作
 */
#include <iostream>
using namespace std;
class Base {
public:
  int m_attrA;
  int m_attrB;
  Base(int attrA, int attrB) {
    m_attrA = attrA;
    m_attrB = attrB;
  }

  Base operator+(Base &p) {
    Base temp(0, 0);
    temp.m_attrA = p.m_attrA + this->m_attrA;
    temp.m_attrB = p.m_attrB + this->m_attrB;
    return temp;
  }
};

ostream &operator<<(ostream &output, Base &base) {
  output << "m_attrA=" << base.m_attrA << " "
         << "m_attrB=" << base.m_attrB << endl;
  return output;
}

void test01() {
  Base A(10, 20);
  Base B(30, 40);
  // 此处等价于调用A.operator+(B)
  Base C = A + B;
  // cout的数据类型为std::ostream
  // cout << "m_attrA=" << C.m_attrA << " "
  // << "m_attrB=" << C.m_attrB << endl;
  cout << A << endl;
}

int main() { test01(); }
