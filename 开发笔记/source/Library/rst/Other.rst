
Other
=====

`GLog <https://github.com/google/glog>`_
--------------------------------------------

CMake
^^^^^

.. code-block:: cmake

   find_package (glog REQUIRED)

   add_executable (myapp main.cpp)
   target_link_libraries (myapp glog::glog)

C++
^^^

.. code-block:: c++

   #include <iostream>
   #include <glog/logging.h>

   int main(int argc, char *argv[]) {
     // Initialize Google’s logging library.
     google::InitGoogleLogging(argv[0]);
     FLAGS_logtostderr = true;
     // Most flags work immediately after updating values.
     LOG(INFO) << "[google] FLAGS_logtostderr";
     FLAGS_logtostderr = false;
   }

Yaml
----

Install
^^^^^^^

.. prompt:: bash $,# auto

   $ sudo apt install libyaml-cpp-dev

CMake
^^^^^

.. code-block:: cmake

   find_package(yaml-cpp REQUIRED)
   target_link_libraries(<可执行文件> ${YAML_CPP_LIBRARIES})
   include_directories(${YAML_CPP_INCLUDE_DIR})

C++
^^^

.. code-block:: c++

   #include "yaml-cpp/yaml.h"

   int main(int argc, const char *argv[]) {

     const std::string file_path = __FILE__;
     const std::string cfg_path =
         file_path.substr(0, file_path.rfind('/')) + "/img_detection.yaml";
     YAML::Node config = YAML::LoadFile(cfg_path);

     const auto classes = config["CLASSES"].as<std::vector<std::string>>();
     const auto config_path = config["CONFIG_PATH"].as<std::string>();
     const auto L1xyR1xy = config["L1xyR1xy"].as<std::vector<std::vector<int>>>();

     return 0;

   }
