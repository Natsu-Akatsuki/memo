# 	Launch

## Glossary

[![image-20210807095158418](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210807095158418.png)](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210807095158418.png)

.. note:: roslaunch在启动节点前，会提前解析 ``substitution args`` 和导入参数(param)到参数服务器

## Reference

- [substitution](http://wiki.ros.org/roslaunch/XML#substitution_args)

- [if and unless attributes](http://wiki.ros.org/roslaunch/XML#if_and_unless_attributes)：所有的tag都支持 ``if`` 和 ``unless`` 属性

- [tag reference](http://wiki.ros.org/roslaunch/XML#Tag_Reference)

## Convention

1. 代码格式化：vscode中使用 `XML Formatter` 进行格式化，缩进为4空格

2. 尽量使用使用substitution标签

3. 添加注释和使用docs属性来描述信息

[![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/xUZKgvoo1W7666ia.png!thumbnail)](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/xUZKgvoo1W7666ia.png!thumbnail)

## Extension

* vscode插件：ROS snippets（代码块/live template）
* vocode插件：Xml Formatter（格式化xml文件）
