# LanguageManage

## 查看当前语系信息

```bash
# 查看当前终端的语系信息
$ echo ${LANG}
$ locale
# 查看当前系统的语系信息
$ localectl
# 查看当前系统支持语系
$ locale -a
# e.g. en_US.UTF-8/ zh_CN.UTF-8
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220511165556136.png" alt="image-20220511165556136" style="zoom:50%;" />

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/vwZa6waF2KX9SxJd.png!thumbnail" alt="img" style="zoom:50%;" />

## [修改当前语系](https://lintut.com/how-to-set-up-system-locale-on-ubuntu-18-04/)

```bash
# >>> 修改系统语系 >>>
# 方法一：直接修改配置文件(/etc/default/locale)
$ update-locale # 可以通过该命令行进行生成
# 方法二：通过命令行修改配置文件 e.g.
$ localectl set-locale LANG=en_US.UTF-8

# >>> 修改终端语系 >>>
# 方法一：通过环境变量进行修改
$ export ...
# 方法二：通过GUI设置(for KDE)，此处为影响终端的语系，不会影响系统的语系（配置文件/etc/default/locale不会被改动）
```

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/ITcSEtbaelh0YHur.png!thumbnail)

.. note:: 生效的最低限度为注销

## 实战

### tty界面不支持中文

- tty界面不支持中文这样复杂的编码文字，因此需要安装一些中文界面的软件

```bash
$ sudo apt install zhcon
$ zhcon --utf8
```
