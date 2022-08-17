#include <cassert>
#include <csignal>
#include <iostream>
void keyboard_handler(int sig) {
  std::cout << "handle keyboard interrupt" << std::endl;
  if (sig == SIGINT)
    ;
  // todo
}

int main(int argc, char **argv) {
  signal(SIGINT /*2*/, keyboard_handler);
  while (1) {
    ;
  }
  return 0;
}
