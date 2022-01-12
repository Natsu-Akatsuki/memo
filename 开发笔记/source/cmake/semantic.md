# Semantic

## 参考资料

* [github例程](https://github.com/ttroy50/cmake-examples)
* [定义和术语](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html)
* [官网](https://cmake.org/cmake/help/latest/index.html)

## 语法

* 指令大小写无关（如 `add_library` 等价于 `ADD_LIBRARY`）

* 参数和变量、OPTION大小写敏感（如REQUIRED不能写成required）

### Alias target

* `target`，根据上下文，应该指的是`library`这种target，而不是executable file；且是 `alias target` ；采用 `taget-based` 的方法可以不用再`include_directory` ，只需要 `target_link_libraries` 就能完成编译（[link](https://github.com/ttroy50/cmake-examples/tree/master/01-basic/H-third-party-library)）

* 案例1：

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/wbtoJSQAxXyl23X8.png!thumbnail)

等价于

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/X74TytKWlvFw0Xst.png!thumbnail)

* [案例2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2/issues/39)

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/srnzrPDtnm75OZuv.png!thumbnail)

* target-based的target（library）采用的是alias targets，其生成的方式可参考 [cmake-examples](https://github.com/ttroy50/cmake-examples/blob/master/01-basic/D-shared-library/README.adoc), [Fexui example](https://github.com/ArthurSonzogni/FTXUI/blob/master/cmake/ftxui_export.cmake)

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/uK5A6MiUUP6Ylf96.png!thumbnail)

## 常用函数/宏

### 变量设置和引用

```cmake
set(SOURCES
    src/Hello.cpp
    src/main.cpp
)
message(${SOURCES})   # src/Hello.cppsrc/main.cpp
set(env{变量名} 值)    # 获取环境变量 
message($env{HOME})   # 使用环境变量
```

.. hint:: 单个variable有多个arguments时，用分号将argument进行concatenate后再进行赋值；然而message显示时，不会出现分号；使用一个变量时，不同于 bash可以不加上{}，在 CMakelists中一定要加上

### 生成库

构建的 `library` 可以不写全名字，如只写hello，cmake会自动补全为 `libhello.so` 或 `libhello.a`

```cmake
add_library(target_name STATIC 文件名)     # 静态库
add_library(target_name SHARED 文件名)     # 动态库
```

### 生成可执行文件

```cmake
add_executable(target_name 文件名)
```

### 添加头文件搜索路径

.. attention:: 只有添加文件夹的，没有直接添加头文件绝对路径的

方法一：

```cmake
# 当前包的头文件目录要放在前面
include_directories(
 include  # 相对于当前CMakeLists所在的文件目录
 ${catkin_INCLUDE_DIRS}
)
```

* 控制追加的路径是放在原来的前面还是后面（设置参数 ON）

```cmake
set(cmake_include_directorirs_before ON)
set(cmake_include_directorirs_after ON)
```

.. hint::  ``#include <file.h>`` 时对应的位置是相对于 ``include_directories`` 中导入的路径例如： ``include_directories`` 的路径是/include/；头文件在/include/package_name/header.h则最终的编写应为 ``#include <package_name/header.h>``

方法二：

```cmake
target_include_directories（target_name
    PUBLIC
        头文件目录）
```

方法二的头文件路径仅适用特定的 `target` ，方法一的适用于所有 `target`

### [添加库搜索路径](https://cmake.org/cmake/help/latest/command/link_directories.html)

```cmake
link_directories(dir_path)
```

.. note:: link_directory只是添加搜索路径，并不起链接作用

### 找库

```cmake
# find_library (<VAR> name1 [path1 path2 ...])
find_library(NVPARSERS NAMES nvparsers)
find_library(NVCAFFE_PARSER NAMES nvcaffe_parser)
find_library(NVINFER_PLUGIN NAMES nvinfer_plugin)
```

* 要添加搜索路径，可修改`CMAKE_LIBRARY_PATH`

```cmake
# e.g.
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "$ENV{HOME}/application/TensorRT-8.0.0.3/lib")
```

### 链接可执行文件与库

```cmake
target_link_libraries(target_name library_name)
```

* [有关关键词option： private、public、target的区别](%5Bhttps://leimao.github.io/blog/CMake-Public-Private-Interface/%5D(https://leimao.github.io/blog/CMake-Public-Private-Interface/))：

本质是用于描述一个链接是否能被继承

* `private`(default)：目标文件A所链接过的库不会被目标文件B 继承
* `public`：目标文件A所链接过的库可被目标文件B 继承
* `interface`：目标文件A所链接过的库不可被目标文件B继承，但是目标文件C链接B时可链接到目标文件A的链接库

有如下案例：比如给定三个文件，分别为可执行文件A `eat_apple` ；库A `fruit` (有size和color两个函数)；库B `apple_libraries` (有apple_size这个函数，该实现依赖 `fruit库` 调用了size函数) 。如果，在实现库B到库A的link时，采用private关键词；eat_apple中调用了apple_size这个函数，依赖了apple_libraries库。此时可执行文件A到库B的link无论使用哪种关键词，都会有link错误。因为前面采用了private关键词，库A到库B的link不会被可执行文件A继承。需要将库B到库A的privatelink改为public link才行。

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/GVwiCAlL2biYLEkP.png!thumbnail)

### [修改target属性](https://cmake.org/cmake/help/v3.18/manual/cmake-properties.7.html#target-properties)

```cmake
set_target_properties(target1 target2 ...
                      PROPERTIES prop1 value1
                      prop2 value2 ...)
```

* 修改文件生成名前/后缀

```cmake
set_target_properties(lib_cpp PROPERTIES PREFIX "")               #  指定前缀
set_target_properties(lib_cpp PROPERTIES OUTPUT_NAME "lib_cpp")   #  指定文件名
set_target_properties(lib_cpp PROPERTIES SUFFIX ".so")            #  指定后缀
set_target_properties(lib_cpp PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})  # 指定库的输出路径
set_target_properties(lib_cpp PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})  # 指定可执行文件的输出路径
```

原来默认生成 `lib_cpp.cpython-37m-x86_64-linux-gnu.so` 现在是 `lib_cpp.so` ；更多属性配置可参考[link](https://cmake.org/cmake/help/latest/manual/cmake-properties.7.html#target-properties)

### [指明链接依赖](https://cmake.org/cmake/help/latest/command/add_dependencies.html)

* 指明在target生成前，需要先生成某些`target`

```cmake
add_dependencies(<target> [<target-dependency>]...)
```

### [给target添加编译选项](https://cmake.org/cmake/help/latest/command/target_compile_options.html?highlight=target_compile_options)

```cmake
# e.g.
add_compile_options(-std=c++14 -O3)
target_compile_options(<target_name> PUBLIC "-g")
# 保留中间产物
target_compile_options(<target_name> PUBLIC "-save-temps")
```

.. note:: 该选项会覆盖CMAKE_BUILD_TYPE

### [find_package](https://cmake.org/cmake/help/v3.18/command/find_package.html?highlight=find_package)

```plain
find_package(<PackageName> [version] [EXACT] [QUIET] [MODULE]
             [REQUIRED] [[COMPONENTS] [components...]]
             [OPTIONAL_COMPONENTS components...]
             [NAMES name1 [name2 ...]]  
             # If the NAMES option is given the names following it are used instead of <PackageName>
             [NO_POLICY_SCOPE])
```

* 指定路径

```cmake
find_package(PCL REQUIRED 
PATHS  库路径
NO_DEFAULT_PATH)  # 只在PATHS路径下寻找，不使用默认的搜索路径
```

* `find_package`宏执行后会产生相关的变量，例如，`<package_name>_INCLUDE_DIRS`
    or `<package_name>_INCLUDES` or
    `<package_name>_INCLUDE_DIR` 具体看相关模块的设计

* cmake modules 指文件`FindXXX.cmake`，要指定 cmake
    module的搜索路径，可以配置如下参数；不过它也有默认的搜索路径即cmake安装路径下的Module目录（e.g.
    /usr/share/cmake-3.16/Modules），在默认路径下没找到，才去CMAKE_MODULE_PATH下找

```cmake
set(CMAKE_MODULE_PATH 路径名)
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/usr/share/cmake/geographiclib/")
```

* `find_packaege`还有一种`Config Mode`，当没找到`FindXXX.cmake`时将按特定的规则进行搜寻，具体可参考[英文文档](https://cmake.org/cmake/help/latest/command/find_package.html#search-procedure)和[中文说明](https://zhuanlan.zhihu.com/p/50829542)（PATH环境变量也会起作用），
* 该种模式下找的是`... LibConfig.cmake`或`...lib_config.cmake`。可添加的搜索路径为

```plain
<package>_DIR
CMAKE_PREFIX_PATH
CMAKE_FRAMEWORK_PATH
CMAKE_APPBUNDLE_PATH
PATH
```

* `find_package`中如果find的包是`catkin`，则`components`用于将`components`涉及的包的环境变量都统一到`catkin_ prefix`的环境变量中。[用与节省敲代码的时间(typing time)](http://wiki.ros.org/catkin/CMakeLists.txt#Why_Are_Catkin_Packages_Specified_as_Components.3F)

### [打印信息](https://cmake.org/cmake/help/latest/command/message.html)

```cmake
message(STATUS|WARNING|FATAL|SEND_ERROR ${})# 这种形式一定要加STATUS这些option
message("...")
```

### catkin_package (ros)

* [官方文档 wiki](http://wiki.ros.org/catkin/CMakeLists.txt#catkin_package.28.29)、[官方文档 api](https://docs.ros.org/en/groovy/api/catkin/html/dev_guide/generated_cmake_api.html#catkin_package)

* 作用：安装`package.xml`；生成可被其他package调用的配置文件(即.config或.cmake文件)。使其他包`find_package`时可以获取这个包的相关信息，如依赖的头文件、库、CMake变量

```cmake
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS cloud_msgs
  DEPENDS PCL
)
add_executable(imageProjection src/imageProjection.cpp)
add_executable(featureAssociation src/featureAssociation.cpp)
add_executable(mapOptmization src/mapOptmization.cpp)
add_executable(transformFusion src/transformFusion.cpp)
```

* 实测其并不会将当前的include等文件夹拷贝到devel目录中
* 必须要在声明targets前（即使用add_library()或add_executable(). 前）
    调用该宏

### [option](https://cmake.org/cmake/help/v3.20/command/option.html)

```cmake
option(<variable> "<help_text>" [value])
```

* 拓展：[option和set的区别？](https://stackoverflow.com/questions/36358217/what-is-the-difference-between-option-and-set-cache-bool-for-a-cmake-variabl)，option只能布尔型，默认是OFF；某些场景下可以相互替换

### 安装

* 可以安装的内容：编译产生的target文件（即可执行文件、库文件）；其他文件
* 若要指定安装路径：

方法一：命令行

```bash
cmake .. -DCMAKE_INSTALL_PREFIX=/install/location
```

方法二：cmake-gui等图形界面进行：

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/fCeDn3uR7Aeffvas.png!thumbnail)

* 指定安装的内容和相对路径：
    安装可执行文件，并安装到到指定目录： `${CMAKE_INSTALL_PREFIX}/bin`

```cmake
install (TARGETS <target_name>
    DESTINATION bin)
```

 安装库文件，并安装到指定目录： `${CMAKE_INSTALL_PREFIX}/lib`

```cmake
install (TARGETS <target_name>
    LIBRARY DESTINATION lib)
```

 安装头文件（即把整个目录拷贝过去）

```cmake
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION include)
```

 安装配置文件，拷贝到 `${CMAKE_INSTALL_PREFIX}/etc`

```cmake
install (FILES <file_name>
    DESTINATION etc)
```

* `make install`后 CMake 会生成
    install_manifest.txt文件（含安装的文件路径，到时可基于这个文件删除安装文件）

```cmake
e.g.
/usr/local/include/ceres/autodiff_cost_function.h
/usr/local/include/ceres/autodiff_first_order_function.h
/usr/local/include/ceres/autodiff_local_parameterization.h
```

.. hint:: 默认安装路径：/usr/local/include; /usr/local/bin; /usr/local/lib/cmake

### 导入额外的CMAKE代码

* 方法一：

```cmake
include(<file|module> [OPTIONAL] [RESULT_VARIABLE <var>]
                      [NO_POLICY_SCOPE])
```

从某个**文件**(CMakeLists.txt)或模块(.cmake)中导入cmake代码；未指定地址时，首先在内置的模块库目录下寻找( `CMake builtin module directory` )，其次在**CMAKE_MODULE_PATH**中寻找

```cmake
set(VTK_CMAKE_DIR "${VTK_SOURCE_DIR}/CMake")
set(CMAKE_MODULE_PATH ${VTK_CMAKE_DIR} ${CMAKE_MODULE_PATH})
include(vtkCompilerChecks)  # /VTK-8.2.0/CMake/vtkCompilerChecks.cmake
```

* 方法二：导入CMakeLists.txt，source_dir对应CMakeLists.txt的所在**目录**

```cmake
add_subdirectory(source_dir [binary_dir] [EXCLUDE_FROM_ALL])
```

### [执行命令行](https://blog.csdn.net/qq_28584889/article/details/97758450)

```cmake
# 相关待执行的命令； 存储标准输出的变量
execute_process(COMMAND python -c "from sysconfig import get_paths;print(get_paths()['include'])" OUTPUT_VARIABLE DUMMY)
```

### 使用通配符找文件

```cmake
# e.g. file(GLOB source_files ${TENSORRT_INSTALL_DIR}/samples/common/*.cpp)
file(GLOB <outPUT-var> [<globbing-expr>...])
```

### 自定义函数

```cmake
# abstract from https://github.com/tier4/AutowareArchitectureProposal.iv/blob/use-autoware-auto-msgs/perception/object_recognition/detection/lidar_centerpoint/CMakeLists.txt

function(download FILE_NAME GFILE_ID FILE_HASH)
# https://drive.google.com/file/d/GFILE_ID/view
message(STATUS "Checking and downloading ${FILE_NAME}")
set(FILE_PATH ${DATA_PATH}/${FILE_NAME})
if(EXISTS ${FILE_PATH})
    file(MD5 ${FILE_PATH} EXISTING_FILE_HASH)
    if(NOT ${FILE_HASH} EQUAL ${EXISTING_FILE_HASH})
    message(STATUS "... file hash changes. Downloading now ...")
    execute_process(COMMAND gdown --quiet https://drive.google.com//uc?id=${GFILE_ID} -O ${FILE_PATH})
    endif()
else()
    message(STATUS "... file doesn't exists. Downloading now ...")
    execute_process(COMMAND gdown --quiet https://drive.google.com//uc?id=${GFILE_ID} -O ${FILE_PATH})
endif()
endfunction()

# default model
download(pts_voxel_encoder_default.onnx 1_8OCQmrPm_R4ZVh70QsS9HZo6uGrlbgz 01b860612e497591c4375d90dff61ef7)
```

## Module CheatSheet

### EIGEN

```cmake
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS} )
```

### OpenCV

```cmake
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(<target> ${OpenCV_LIBS})
```

## Variables CheatSheet

### python

[FindPythonLibs](https://cmake.org/cmake/help/v3.10/module/FindPythonLibs.html) / [FindPythonInterp](https://cmake.org/cmake/help/v3.10/module/FindPythonInterp.html?highlight=python_executable)

```bash
-DPYTHON_EXECUTABLE=/opt/conda/bin/python3
-DPYTHON_EXECUTABLE=$(python -c "import sys;print(sys.executable)")

-DPYTHON_INCLUDE_DIR=$(python -c "from sysconfig import get_paths;print(get_paths()['include'])")
-DPYTHON_LIBRARY=/opt/conda/lib/libpython3.8.so

-DPYBIND11_PYTHON_VERSION=3.7
-DPYTHON_VERSION=3.7
```

### compiler

```bash
# 指定使用c++14标准
set(CMAKE_CXX_FLAGS "-std=c++14")
```

### [ros](http://docs.ros.org/en/kinetic/api/catkin/html/user_guide/variables.html)

### path

|                           Variable                           |                             Info                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                       CMAKE_SOURCE_DIR                       |                  The root source directory                   |
|                   CMAKE_CURRENT_SOURCE_DIR                   | The current source directory **if using sub-projects and directories**. |
|                      PROJECT_SOURCE_DIR                      |      The source directory of the current cmake project.      |
|                       CMAKE_BINARY_DIR                       | The root binary / build directory. This is the directory where you ran the cmake command. |
|                   CMAKE_CURRENT_BINARY_DIR                   |          The build directory you are currently in.           |
|                      PROJECT_BINARY_DIR                      |         The build directory for the current project.         |
| [LIBRARY_OUTPUT_PATH](https://cmake.org/cmake/help/v3.18/variable/LIBRARY_OUTPUT_PATH.html?highlight=library_output_path) (deprecated)LIBRARY_OUTPUT_DIRECTORY |           库的输出路径（要设置在add_library之前）            |
|                      CMAKE_PREFIX_PATH                       |    find_packaeg 搜索.cmake .config的搜索路径（初始为空）     |
|                    EXECUTABLE_OUTPUT_PATH                    |                     可执行文件的输出路径                     |
