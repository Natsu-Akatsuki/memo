#include "NvInferRuntime.h"
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

class Logger : public nvinfer1::ILogger {
public:

  void log(Severity severity, const char *msg) override {
    // 设置日志等级
    if (severity <= Severity::kWARNING) {
      timePrefix();
      std::cout << severityPrefix(severity) << std::string(msg) << std::endl;
    }
  }

private:
  static const char *severityPrefix(Severity severity) {
    switch (severity) {
    case Severity::kINTERNAL_ERROR:
      return "[F] ";
    case Severity::kERROR:
      return "[E] ";
    case Severity::kWARNING:
      return "[W] ";
    case Severity::kINFO:
      return "[I] ";
    case Severity::kVERBOSE:
      return "[V] ";
    default:
      // #include <cassert>
      assert(0);
      return "";
    }
  }
  void timePrefix() {
    std::time_t timestamp = std::time(nullptr);
    tm *tm_local = std::localtime(&timestamp);
    std::cout << "[";
    std::cout << std::setw(2) << std::setfill('0') << 1 + tm_local->tm_mon
              << "/";
    std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_mday << "/";
    std::cout << std::setw(4) << std::setfill('0') << 1900 + tm_local->tm_year
              << "-";
    std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_hour << ":";
    std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_min << ":";
    std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_sec << "] ";
  }
} logger;

int main() {

  logger.log(nvinfer1::ILogger::Severity::kVERBOSE, "ouput VERBOSE message");
  logger.log(Logger::Severity::kINFO, "ouput INFO message");
  logger.log(Logger::Severity::kWARNING, "ouput WARNING message");
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  logger.log(Logger::Severity::kERROR, "ouput ERROR message");
  logger.log(Logger::Severity::kINTERNAL_ERROR, "ouput INTERNAL_ERROR message");

  return 0;
}