

- 此处的`commmit` 等价于`git仓`或`history`



## 压缩commit记录

- （目的）减小git仓的大小、去冗余、让commit记录更漂亮



 

## 修改message (of last commit)

- 在本地文件内容 = 暂存区内容 = 本地仓内容时

``` bash
$ git commit --amend -m "<修改后的message>"
```

- pycharm IDE中的实现：

  <img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210209114613082.png" alt="image-20210209114613082" style="zoom: 67%;" />



## 修改远程服务器上的git仓（暴力解决方案）

- 在本地修正完本地仓的历史后，强制将本地仓的历史覆写到远程仓中

```bash
$ git push -f 
```



## 选择性地挑选文件的changes进行commit

![image-20210222010451820](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210222010451820.png)





## 版本回溯

```bash
# 回溯到对应的commit
$ git reset [option] [commit]
soft  ：只修改本地仓 HEAD 的指向
mixed ：令暂存区与git仓同步
hard  ：在mixed的基础上，让同步工作目录
```



## 删除commit

```bash
# 删除文件在暂存区和工作区的相关信息
$ git rm <文件/文件夹>
# 只删除其在暂缓区的相关信息 
$ git rm --cached <文件/文件夹
```

