## [Packaging](https://setuptools.pypa.io/en/latest/userguide/index.html)

- pip等为packaging的前端；setuptools则为后端
- 目前的规范为：工程的元数据都放在`pyproject.toml`文件

### CLI

```bash
# 等价于pip install -e . (可编辑模型/开发模型：需要频繁地改动，本地的修改执行反应到安装的包上)
$ python setup.py develop
# 删除所有编译文件(build目录)
$ python setup.py clean --all

# build
$ python -m build

# install
$ pip install dist/...whl
$ pip install dist/...tar.gz
```

---

**NOTE**

* [AttributeError: install_layout](https://stackoverflow.com/questions/36296134/attributeerror-install-layout-when-attempting-to-install-a-package-in-a-virtual)：更新setuptools

---

### Usage

#### 简例

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

#### 重写命令行

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

#### 指定安装的依赖

```python
setup(
 install_requires=[
    'rospkg==0.5.0'
        'numpy>=0.3.0',
        'setuptools>=1.0.0,<2.0.0'
 ]
)
```

#### [添加pip包的url](https://peps.python.org/pep-0633/)

#### [指定package的搜索位置](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html)

### Reference

* [setup 关键字的中文解析](https://www.cnblogs.com/xueweihan/p/12030457.html)
* [简书 教程](http://www.smartredirect.de/redir/clickGate.php?u=IgKHHLBT&m=1&p=8vZ5ugFkSx&t=vHbSdnLT&st=&s=&url=http%3A%2F%2Fwww.smartredirect.de%2Fredir%2FclickGate.php%3Fu%3DIgKHHLBT%26m%3D1%26p%3D8vZ5ugFkSx%26t%3DvHbSdnLT%26st%3D%26s%3D%26url%3Dhttps%3A%2F%2Fwww.jianshu.com%2Fp%2F9a5e7c935273%26r%3Dhttps%3A%2F%2Fshimo.im%2Fdocs%2FgK6WtttVjdytQgCX&r=https%3A%2F%2Fshimo.im%2Fdocs%2FgK6WtttVjdytQgCX)
* [案例：PointCloud-PyUsage](https://github.com/Natsu-Akatsuki/PointCloud-PyUsage)
