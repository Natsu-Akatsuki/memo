## logging

### 简例

```python
import logging

logging.debug("This is a debug log.")
logging.info("This is a info log.")
logging.warning("This is a warning log.")
logging.error("This is a error log.")
logging.critical("This is a critical log.")

>>> WARNING:root:This is a warning log.
>>> ERROR:root:This is a error log.
>>> CRITICAL:root:This is a critical log.
```

- 默认情况只有warning以上等级的日志才会被输出
- 每行输出的字段含义为：日志级别:日志器名称:日志内容

### 实用资料

- [实例和用法解析(cnblogs)](https://www.cnblogs.com/yyds/p/6901864.html)

- [支持的格式化属性](https://docs.python.org/3/library/logging.html#logrecord-attributes)

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210907230454217.png" alt="image-20210907230454217" style="zoom:67%;" />

## multithreading

- [native_id和identity的区别？](https://docs.python.org/3/library/threading.html#threading.get_ident)

前者是操作系统对线程的标识号，后者是python的标识号