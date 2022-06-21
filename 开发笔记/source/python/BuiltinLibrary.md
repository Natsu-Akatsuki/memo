# BuiltinLibrary

## collections

## [defaultdict](https://docs.python.org/3/library/collections.html#defaultdict-objects)

* [defaultdict和dict的区别？](https://www.jianshu.com/p/bbd258f99fd3)（没键时会返回工厂函数默认值）

## [logging](https://docs.python.org/3/library/logging.html)

* [基本使用](https://www.cnblogs.com/yyds/p/6901864.html)

```python
import logging
# 自定义logging显示等级和前缀
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logging.debug("This is a debug log.")
logging.info("This is a info log.")
logging.warning("This is a warning log.")
logging.error("This is a error log.")
logging.critical("This is a critical log.")

# 2022-03-05 10:48:29,309 [INFO] This is a info log.
# 2022-03-05 10:48:29,309 [WARNING] This is a warning log.
# 2022-03-05 10:48:29,309 [ERROR] This is a error log.
# 2022-03-05 10:48:29,309 [CRITICAL] This is a critical log.
```

## [multiprocessing](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing)

* Cpython不是线程安全的 ，因此需要使用GIL；GIL即一个`互斥` (mutex)：能确保解释器一次只能执行某个线程的python字节码

### 进程的若干种状态

* 进程start时，状态为initial

* 进程start后，状态为started

* 执行完run的内容后，状态为stopped，含退出码(exit code)

#### python多进程中定义信号处理函数、自定义进程类

```python
class RosLaunchProc(Process):
    def __init__(self, args):
        """创建一个进程来启动roslaunch文件"""
        super().__init__()
        self.launch_proc = None
        self.roslaunch_files = args[0]
        self.is_core = args[1]

    def shutdown_handle(self, sig, frame):
        """
        自定义信号回调函数调用shutdown，调用后将disable spin，使进程terminate
        """
        self.launch_proc.shutdown()
        rospy.loginfo(f"\033[1;31m成功调用launch.shutdown()\033[0m")

    def run(self):
        # 信号函数的register需要放在run（i.e.主线程）
        signal.signal(signal.SIGUSR1, self.shutdown_handle)
        self.launch_proc = self.roslaunch_api()
        self.launch_proc.start()
        # 阻塞，防止进程stop状态
        rospy.spin()

    def roslaunch_api(self):
        uuid = roslaunch.rlutil.get_or_generate_uuid(options_runid=None, options_wait_for_master=False)
        roslaunch.configure_logging(uuid)
        return roslaunch.parent.ROSLaunchParent(uuid, self.roslaunch_files, self.is_core)
```

#### 关闭进程

```python
# 释放进程对象和与之相关的资源，在close前应该terminate()关闭该进程/或该进程已经stopped
<process_obj>.terminate()
<process_obj>.close()
```

#### 将某个函数放到新进程执行

```python
import time
from multiprocessing import Process
import os

def f():
    print(f'subProcess: {os.getpid()}')

if __name__ == '__main__':
    p = Process(target=f)
    p.start()
    time.sleep(1)
    print(f'fatherProcess: {os.getpid()}')
```

## os

### 设置环境变量

```python
import os

# 设置环境变量
os.environ["..."] = "value"
# 获取环境变量
os.getenv("环境变量名")
```

## signal

```python
def handle_int(sig, frame):
    """
    自定义信号回调函数
    Returns:

    """
    print("get signal: %s, I will quit" % sig)
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(2, handle_int)
```

## [shutil](https://docs.python.org/3/library/shutil.html#)

```bash
# 文件拷贝
shutil.copy(f"str <src文件>", "str <dst文件夹/文件>")
# cp -r 文件夹拷贝
shutil.copytree()
```

## [struct](https://docs.python.org/3/library/struct.html)

将python value转为C struct(在python中struct为`bytes object`)

### 返回一个字节对象

struck.pack(\<format>,value...)

```python
>>> from struct import *
# 返回一个C结构体(用字节对象来表征)
>>> pack('hhl', 1, 2, 3)
b'\x00\x01\x00\x02\x00\x00\x00\x03'
>>> unpack('hhl', b'\x00\x01\x00\x02\x00\x00\x00\x03')
(1, 2, 3)
>>> calcsize('hhl')
8
```

<https://docs.python.org/3/reference/lexical_analysis.html#string-and-bytes-literals>

## 类型检测

```python
# 判断某个对象是否某个类的实例
isinstance(value, int)
isinstance(value, np.ndarray)
```

## 属性操作

```python
# 取值
getattr(实例, 属性名） # 等价于 实例.属性
# __getattribute__（有无该值，都会调用该函数）
# __getattr__(没有该值时，则调用该函数)

# 赋值
setattr(实例，属性名, value) # 等价于 实例.属性 = value

# 判断
hasattr（判断某值是否存在）
```

## 字符串操作

```python
# 计算某个字符的出现次数
<str>.count(<sub_str>) 
# 字符串分割（返回列表）
<str>.split(<sub_str>)
```

---

**NOTE**

提取子串的如下操作会返回空值

```python
str = "AB"
strA = str[1:-1]  # []
```

---

## [字典操作](https://docs.python.org/3/library/stdtypes.html?highlight=dict#mapping-types-dict)

* 字典会保留插入顺序

```python
# 构建字典
d = {"one": 1, "two": 2, "three": 3, "four": 4}
```

### 键值对操作

```python
# 返回字典的key列表
list(d)
# 返回字典的value列表
list(d.values())
[1, 2, 3, 4]
# 更新值
d["one"] = 42
# 删除某个键值对
del d["two"]
```

## [subprocess](https://docs.python.org/3.7/library/subprocess.html)

### subprocess.call

```python
# 父进程会等子进程完成，有返回值exitcode, 在终端有输出结果
subprocess.call("cmd", shell=True)
# checkcall     效果类似，只是返回值不为0时会抛出异常（有标准输出错误/标准输出）
# check_output  同check_call，但终端无输出结果，返回值为终端输出结果
# run           返回一个CompletedProcess对象，终端有输出结果

# option:
# cwd: <change working directory 路径跳转，此为执行命令的路径，可为相对路径>
# env: <环境变量>
```

### Q&A

* os.system和subprocess的区别？([ref](https://docs.python.org/3/library/subprocess.html#replacing-os-system))

后者是前者的超集，可更自定义和灵活（能处理SIGINT和SIGQUIT信号）

```bash
sts = os.system("mycmd" + " myarg")
# becomes
retcode = call("mycmd" + " myarg", shell=True)
```

## [setuptools](https://setuptools.pypa.io/en/latest/userguide/index.html)

### 命令行

```bash
# 等价于pip install -e . (可编辑模型/开发模型：需要频繁地改动，本地的修改执行反应到安装的包上)
$ python setup.py develop
# 删除所有编译文件(build目录)
$ python setup.py clean --all
```

---

**NOTE**

* [AttributeError: install_layout](https://stackoverflow.com/questions/36296134/attributeerror-install-layout-when-attempting-to-install-a-package-in-a-virtual)：更新setuptools

---

### usage

* 简例

```python
from setuptools import find_packages, setup
setup(
    name='包名',
    version='0.0.1',
    packages=['ros_numpy'],  # 指定要打包的包，或者由程序find_package()寻找
    author='Natsu_Akatsuki',
    description='...'
)
```

* find_package返回的是module list

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/meeAd0u0LQ1Nfucc.png!thumbnail)

* 重写命令行

```python
import shutil
from distutils.command.clean import clean
from pathlib import Path
from distutils.cmd import Command
from setuptools import find_packages, setup
import os

package_name = "am-utils"


class UninstallCommand(Command):
    description = "uninstall the package and remove the egg-info dir"
    user_options = []

    # This method must be implemented
    def initialize_options(self):
        pass

    # This method must be implemented
    def finalize_options(self):
        pass

    def run(self):
        os.system("pip uninstall -y " + package_name)
        dirs = list((Path('.').glob('*.egg-info')))
        if len(dirs) == 0:
            print('No egg-info files found. Nothing to remove.')
            return

        for egg_dir in dirs:
            shutil.rmtree(str(egg_dir.resolve()))
            print(f"Removing dist directory: {str(egg_dir)}")


class CleanCommand(clean):
    """
    Custom implementation of ``clean`` setuptools command."""

    def run(self):
        """After calling the super class implementation, this function removes
        the dist directory if it exists."""
        self.all = True  # --all by default when cleaning
        super().run()
        if Path('dist').exists():
            shutil.rmtree('dist')
            print("removing 'dist' (and everything under it)")
        else:
            print("'dist' does not exist -- can't clean it")


setup(
    name=package_name,
    packages=find_packages(),
    author='anomynous',
    cmdclass={'uninstall': UninstallCommand, # 重写命令行选项
              'clean': CleanCommand},
)

```

* 指定安装的依赖

```python
setup(
 install_requires=[
    'rospkg==0.5.0'
        'numpy>=0.3.0',
        'setuptools>=1.0.0,<2.0.0'
 ]
)
```

### 拓展资料

* [setup 关键字的中文解析](https://www.cnblogs.com/xueweihan/p/12030457.html)

* [简书 教程](http://www.smartredirect.de/redir/clickGate.php?u=IgKHHLBT&m=1&p=8vZ5ugFkSx&t=vHbSdnLT&st=&s=&url=http%3A%2F%2Fwww.smartredirect.de%2Fredir%2FclickGate.php%3Fu%3DIgKHHLBT%26m%3D1%26p%3D8vZ5ugFkSx%26t%3DvHbSdnLT%26st%3D%26s%3D%26url%3Dhttps%3A%2F%2Fwww.jianshu.com%2Fp%2F9a5e7c935273%26r%3Dhttps%3A%2F%2Fshimo.im%2Fdocs%2FgK6WtttVjdytQgCX&r=https%3A%2F%2Fshimo.im%2Fdocs%2FgK6WtttVjdytQgCX)

## time

```python
import time
# measure wall time
start = time.time()
print('TIME(ms)  is=',1000 * (time.time() - start))

# Return the value (in fractional seconds) of the sum of the system and user CPU time of the current process. It does not include time elapsed during sleep. It is process-wide by definition.
# measure process time
start = time.process_time()
print('TIME(ms)  is=',1000 * (time.process_time() - start))
```

### migration

python3.8后已移除time.clock()，可以使用time.perf_counter()或time.process_time()方法替代
