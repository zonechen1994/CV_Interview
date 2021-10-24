# 110道Python面试笔试题汇总

#### 1、一行代码实现1--100之和

利用sum()函数求和

![](https://files.mdnice.com/user/11419/727019db-930a-4907-83a0-4f905ee69d1b.png)

#### 2、如何在一个函数内部修改全局变量

利用global修改全局变量

​                                                         ![](https://files.mdnice.com/user/11419/19277f42-90ef-4f0a-8084-f77db2247fbe.png) 

#### 3、列出5个python标准库

OS：提供了不少与操作系统相关联的函数 

sys:通常用于命令行参数

re:正则匹配

math:数学运算 

datetime:处理日期时间

#### 4、字典如何删除键和合并两个字典

del和update方法

![](https://files.mdnice.com/user/11419/4c299471-d45d-4f05-8520-6fe38d7b877f.png)

#### 5、谈下python的GIL

GIL是python的全局解释器锁，同一进程中假如有多个线程运行，一个线程在运行 python程序的时候会霸占python解释器(加了一把锁即GIL),使该进程内的其他程 无法运行，等该线程运行完后其他线程才能运行。如果线程运行过程中遇到耗时操作，则 解释器锁解开，使其他线程运行。所以在多线程中，线程的运行仍是先后顺序的，并不是同时进行。

多进程中因为每个进程都能被系统分配资源，相当于每个进程有了一个python解释器， 所以多进程可以实现多个进程的同时运行，缺点是进程系统资源开销大

#### 6、python实现列表去重的方法

先通过集合去重，在转列表

![](https://files.mdnice.com/user/11419/9835d496-b9fe-43d4-a1f5-cf6cf89f131f.png)

#### 7、fun(*args,**kwargs)中的**   *args,**kwargs什么意思

*args和  **kwargs主要用于函数定义。你可以将不定数量的参数传递给—函数。这里的不定的意思是:预先并不知道，函数使用者会传递多少个参数给你，所以在这个场景下使用这两个关键字。  *args是用来发送一个非键值对的可变数量的参数列表给一个函数.这里有个倒序帮你理解这个概念：

![](https://files.mdnice.com/user/11419/89f52ed5-6f84-4c60-8662-b2b8ed4b760c.png)

![](https://files.mdnice.com/user/11419/ebff14fd-6ded-41f4-b4cb-afa56a9ccae0.png)

#### 8、Python2和python3的range（100）的区别

python2返回列表，python3返回迭代器，节约内存

#### 9、一句话解释什么样的语言能够用装饰器？

函数可以作为参数传递的语言，可以使用装饰器

#### 10、python内建数据类型有哪些

整型--int

布尔型--bool 

字符串-str 

列表—list 

元组--tuple 

字典--diet

#### 11、简述面向对象中_ new 和 init _ 区别

init是初始化方法，创建对象后，就立刻被默认调用了，可接收参数，如图

![](https://files.mdnice.com/user/11419/674c80cf-1829-452e-8b04-26a9216a2b67.png)

1、__new__至少要有一个参数cis,代表当前类，此参数在实例化时由Python解释器自动识别

2、__new__必须要有返回值，返回实例化出来的实例，这点在自己实现__new__时要特 别注意，可以return父类（通过super（当前类名，cis）） __new__出来的实例，或者直接是object的new出来的实例

3、__init__有一个参数self,就是这个__new__返回的实例，__init__在__new__的基础上可以完成一些其他初始化的动作，__init__不需要返回值。

4、如果__new__创建的是当前类的实例，会自动调用__init__函数，通过return语句里面调用的__new__函数的第一个参数是cis来保证是当前类实例，如果是其他类的类名，; 那么实际创建返回的就是其他类的实例，其实就不会调用当前类的__init__函数，也不会调用其他类的__init__函数。

![](https://files.mdnice.com/user/11419/e8e73fb7-89ac-4f94-afa9-a4e84dfd0347.png)

#### 12、方法打开处理文件帮我们做了什么？



![](https://files.mdnice.com/user/11419/86c96c17-cb13-49af-9042-be0563f85b68.png)

打开文件在进行读写的时候可能会出现一些异常状况，如果按照常规的f.open写法，我们需要try,except,finally,做异常判断，并且文件最终不管遇到什么情况，都 要执行finally f.close()关闭文件，with方法帮我们实现Tfinally中f.close

(当然还有其他自定义功能，有兴趣可以研究with方法源码)

#### 13、列表[1,234,5],请使用map()函数输出[1,4,9,16,25],并使用列表推导式提取出大于10的数，最终输出[16,25]

map ()函数第一个参数是fun,第二个参数是一般是list,第三个参数可以写list,也可以不写，根据需求

![](https://files.mdnice.com/user/11419/f2fd6c03-e97c-475b-af96-059987a4be1d.png)

#### 14、python中生成随机整数、随机小数、0--1之间小数方法

随机整数：random.randint(a,b),生成区间内的整数

随机小数：习惯用numpy库，利用np.random.randn⑸生成5个随机小数 

0-1随机小数：random.random(),括号中不传参

![](https://files.mdnice.com/user/11419/c3392afd-329b-4017-8879-8b8b0971f537.png)



#### 15、避免转义给字符串加哪个字母表示原始字符串?

r,表示需要原始字符串，不转义特殊字符

#### 16、<div class="nam">中国</div>用正则匹配出标签里面的内容（"中国"）, 中class的类名是不确定的

![](https://files.mdnice.com/user/11419/779e454a-86a6-4205-a3d0-09b6102f3a40.png)

#### 17、python中断言方法举例

![](https://files.mdnice.com/user/11419/63e7fc0a-0c01-436c-8644-d7a79f45fb95.png)

#### 18、数据表student有id，name，score，city字段，其中name中的名字可有重复，需要 消除重复行，请写sql语句

select distinct name from student

#### 19、10个linux常用命令

Is pwd cd touch rm mkdir tree cp mv cat more grep echo

#### 20、python2和python3区别？列举5个

1、Python3使用print必须要以小括号包裹打印内容，比如print('hi')

Python2既可以使用带小括号的方式，也可以使用一个空格来分隔打印内容，比如 print 'hi'

2、python2 range(1,10)返回列表，python3中返回迭代器，节约内存

3、python2中使用ascii编码，python中使用utf-8编码

4、python2中unicode表示字符串序列，str表示字节序歹U

python3中str表示字符串序列，byte表示字节序列

5、python2中为正常显示中文，弓|入coding声明，python3中不需要

6、python2中是raw_input()函数，python3中是input()函数

#### 21、列出python中可变数据类型和不可变数据类型，并简述原理

不可变数据类型：数值型、字符串型string和元组tuple

不允许变量的值发生变化，如果改变了变量的值，相当于是新建了一个对象，而对于相同 的值的对象，在内存中则只有一个对象(一个地址),如下图用id()方法可打印对象的 id

![](https://files.mdnice.com/user/11419/b05344a7-7903-42ca-a0f2-dfb40df98d1a.png)

可变数据类型：列表list和字典diet;

允许变量的值发生变化，即如果对变量进行append. +=等这种操作后，只是改变了变 量的值，而不会新建一个对象，变量引用的对象的地址也不会变化，不过对于相同的值的不同对象，在内存中则会存在不同的对象，即每个对象都有自己的地址，相当于内存中对 于同值的对象保存了多份，这里不存在引用计数，是实实在在的对象。

​                                                                        ![](https://files.mdnice.com/user/11419/598ee4ce-5c4e-43d3-be12-3594f6d7d471.png)

 

#### 22、s="ajIdjIajfdIjfddd"去重并从小到大排序输出

set去重，去重转成list,利用sort方法排序，reverse=False是从1倒大排 list是不变数据类型，s.sort时候没有返回值，所以注释的代码写法不正确

​                    ![](https://files.mdnice.com/user/11419/99a95284-2a26-4011-9358-33f2d71b6c96.png)                      

#### 23、用lambda函数实现两个数相乘

dict={"name":"zs","age":18,"city":"深圳","tel":"1362626627"}

![](https://files.mdnice.com/user/11419/a186b04b-e94a-4a9d-852f-dbd872510a05.png)

#### 24、字典根据键从小到大排序

dict={"name":"zs","age":18,":"深圳","tel":"1362626627"}

![](https://files.mdnice.com/user/11419/553df875-82e3-489c-b74e-427ce23891a6.png)



#### 25、利用collections库的Counter方法统计字符串毎个单词出现的次数"kjalfj;ldsjafl;hdsllfdhg;lahfbl;hl;ahlf;h"

![](https://files.mdnice.com/user/11419/79aaab3b-45ed-48a5-9d82-1b04b8b66bec.png)

#### 26、字符串a="not 404 found 张三 99 深圳''，每个词中间是空格，用正则过滤掉英文和数字，最终输出"张三 深圳"

![](https://files.mdnice.com/user/11419/0201ba2b-4ba4-4bc7-a56a-ab7775e27adc.png)

顺便贴上匹配小数的代码，虽然能匹配，但是健壮性有待进一步确认

​                ![](https://files.mdnice.com/user/11419/3bbf9d48-05ee-4c6a-8183-efec6e956899.png) 

#### 27、filter方法求出列表所有奇数并构造新列表，a=[1,2,3,4,5,6,7,8,9,10]

filter()函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。该接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回True或False,最后将返回True的元素放到新列表

![](https://files.mdnice.com/user/11419/89aba4e4-3fbe-4d99-8059-e29333349833.png)

#### 28、列表推导式求列表所有奇数并构造新列表，a=[1,2,3, 4, 5, 6, 7, 8, 9,10]

![](https://files.mdnice.com/user/11419/8b3a27ff-ad83-4081-afde-b79bcc51e252.png)

#### 29、正则re.compile作用

recompile是将正则表达式编译成一个对象，加快速度，并重复使用

#### 30、a=（1，）b=(1),c=("1")分别是什么类型的数据？

![](https://files.mdnice.com/user/11419/0a602557-4ebb-49d3-abd7-81532b96be15.png)

 

#### 31、两个列表［1,5,7,9］和［2,2,6,8］合并为［1,2,2,3,6,7,8,9］

extend可以将另一个集合中的元素逐一添加到列表中，区别于append整体添加

![](https://files.mdnice.com/user/11419/e50a01e9-acac-4f9c-8b76-bd031aaf3fca.png)



#### 32、用python删除文件和用Linux命令删除文件方法

python: os.remove(文件名)

linux:	rm文件名

#### 33、log日志中，我们需要用时间戳记录error,warning等的发生时间，请用datetime模块打印当前时间戳"2018-04-01 11:38:54"

顺便把星期的代码也贴上了

![](https://files.mdnice.com/user/11419/6e6f07bb-8522-4c68-b8a8-e62a973d3ac7.png)

#### 34、数据库优化查询方法

外键、索引、联合查询、选择特定字段等等

#### 35、请列出你会的任意一种统计图(条形图、折线图等)绘制的开源库，第三方也行

pychart、matplotlib

#### 36、写一段自定义异常代码

自定义异常用raise抛出异常

![](https://files.mdnice.com/user/11419/a23f1a2d-32ed-4920-abf6-ad90802c3eda.png)

#### 37、正则表达式匹配中，(.\*)和(.\*?)匹配区别？

(.)是贪婪匹配，会把满足正则的尽可能多的往后匹配

(.?)是非贪婪匹配，会把满足正则的尽可能少匹配

![](https://files.mdnice.com/user/11419/4435ae27-ae8b-4f06-99fd-430568166292.png)

#### 38、简述Django的orm

ORM,全拼Object-Relation Mapping,意为对象-关系映射

实现了数据模型与数据库的解耦，通过简单的配置就可以轻松更换数据库，而不需要修改代码只需要面向对象编程,orm操作本质上会根据对接的数据库引擎，翻译成对应的sql语 句，所有使用Django开发的项目无需关心程序底层使用的是MySQL、Oracle, sqlite....,如果数据库迁移，只需要更换Django的数据库引擎即可

![](https://files.mdnice.com/user/11419/f2d03565-adfb-48b9-bbc8-522fd83be110.png) 

#### 39、[[1,2],[3,4],[5,6]]一行代码展开该列表，得出[1,2,3,4,5,6]

列表推导式的骚操作

运行过程：for i in a，每个i是【1,2】，【3,4】，【5,6】,for j in i,每个j就是 1,2,3,4,5,6,合并后就是结果

![](https://files.mdnice.com/user/11419/eb9d659b-a5d1-4ba2-837c-014a7dcb8a81.png)

还有更骚的方法，将列表转成numpy矩阵，通过numpy的flatten ()方法，代码永远是只有更骚，没有最骚

![](https://files.mdnice.com/user/11419/ed1b2936-8890-47fd-a78f-dd19efaf9f32.png)

#### 40、x="abc",y="def",z=["d”,"e","f"]，分别求出x.join(y)和x.join(z)返回的结果

join。括号里面的是可迭代对象，x插入可迭代对象中间，形成字符串，结果一致，有没 有突然感觉字符串的常见操作都不会玩了

顺便建议大家学下os.path.joinQ方法，拼接路径经常用到，也用到Tjoin,和字符串操作 中的join有什么区别，该问题大家可以査阅相关文档，后期会有答案

![](https://files.mdnice.com/user/11419/1e7658ae-4c22-47b4-814d-a7f3348d09e8.png)

#### 41、举例说明异常模块中try except else finally的相关意义

try..except..else没有捕获到异常，执行else语句 try..except..finally不管是否捕获到异常,都执行finaIly语句

try..except..finally不管是否捕获到异常，都执行finall语句

![](https://files.mdnice.com/user/11419/b614e119-274a-4274-8b56-4354e34aa924.png)

#### 42、python中交换两个数值

​                                                             ![](https://files.mdnice.com/user/11419/691d8ae8-c28b-45ea-acc8-a3447d63bedd.png) 

#### 43、举例说明zip()函数用法

zip()函数在运算时，会以一个或多个序列(可迭代对象)做为参数，返回一个元组的列 表。同时将这些序列中并排的元素配对。

zip()参数可以接受任何类型的序列，同时也可以有两个以上的参数;当传入参数的长度不 同时，zip能自动以最短序列长度为准进行截取，获得元组。

![](https://files.mdnice.com/user/11419/e477c3b6-b64d-4029-bf18-342676c810cc.png)

#### 44、a="张明98分"，用re.sub,将98替换为100

![](https://files.mdnice.com/user/11419/fd603dc0-ebc7-43bc-8d8e-32b9e9859c1b.png)



#### 45、写5条常用sql语句

show databases;

show tables;

desc表名；

select * from 表名；

delete from 表名 where id = 5;

update students set gender=0,hometown = "北京"where id = 5

#### 46、a="hello“和b="你好"编码成bytes类型

![](https://files.mdnice.com/user/11419/c1fc0c92-a485-4f95-8faa-2cdd0054ce85.png)

#### 47、[1,2,3]+ [4,5,6]的结果是多少?

两个列表相加，等价于extend

![](https://files.mdnice.com/user/11419/8ef04363-1811-4287-8a2d-9549f19562fb.png)

#### 48、python运行效率的方法

1、	使用生成器，因为可以节约大量内存

2、	循环代码优化，避免过多重复代码的执行

3、	核心模块用Cython PyPy等，提高效率

4、	多进程、多线程、协程

5、	多个ifelif条件判断，可以把最有可能先发生的条件放到前面写，这样可以减少程序判断的次数，提高效率

#### 49、简述mysql和redis区别

redis:内存型非关系数据库，数据保存在内存中，速度快

mysql:关系型数据库，数据保存在磁盘中，检索的话，会有一定的I。操作，访问速度相对慢

#### 50、遇到bug如何处理

1、细节上的错误，通过print ()打印，能执行到print ()说明一般上面的代码没有问 题，分段检测程序是否有问题，如果是js的话可以alert或console.log

2、如果涉及一些第三方框架，会去查官方文档或者一些技术博客。

3、对于bug的管理与归类总结，一般测试将测试出的bug用teambin等bug管理工具进 行记录，然后我们会一条一条进行修改，修改的过程也是理解业务逻辑和提高自己编程逻 辑缜密性的方法，我也都会收藏做一些笔记记录.

4、导包问题、城市定位多音字造成的显示错误问题

#### 51、正则匹配，匹配日期2018-03-20

url ='https://sycm.taobao.com/bda/tradinganaly/overview/get_summaryjson? dateRange=2018-03-20%7C2018-03-

20&dateType=recent1&device=1&token=ff25b109b&_=15215956134621 仍有同学问正则，其实匹配并不难，提取一段特征语句，用(.*?)匹配即可

![](https://files.mdnice.com/user/11419/2fc5ec2d-a9b8-423f-8a5d-b0c3c74657a2.png)

#### 52、list=[2,3,5,4,9,6],从小到大排序，不许用sort,输出[2,3,4,5,6,9]

利用min。方法求出最小值，原列表删除最小值，新列表加入最小值，递归调用获取最小值的函数，反复操作

![](https://files.mdnice.com/user/11419/0ec8eb61-e623-45da-bd30-2aff79ef518e.png)

#### 53、写一个单列模式

因为创建对象时_new_方法执行，并且必须return返回实例化出来的对象所 cls._instance是否存在，不存在的话就创建对象，存在的话就返回该对象，来保证只有一个实例对象存在（单列），打印ID,值一样，说明对象同一个

![](https://files.mdnice.com/user/11419/ceece4dd-9f28-4c39-ac00-19d0eeb3685e.png)

#### 54、保留两位小数

题目本身只有a = "%.03f"%1.3335,让计算a的结果，为了扩充保留小数的思路，提供 round方法(数值，保留位数)

![](https://files.mdnice.com/user/11419/40b77d8b-8a07-46c3-bf6d-a754f473ba53.png)

#### 55、求三个方法打印结果

fn("one",1)直接将键值对传给字典；

fn("two",2)因为字典在内存中是可变数据类型，所以指向同一个地址，传了新的额参数后，会相当于给字典增加键值对

fn("three",3,{})因为传了一个新字典，所以不再是原先默认参数的字典

![](https://files.mdnice.com/user/11419/1ccec14c-cf44-4506-ac24-de0f1da476c2.png)

#### 56、列出常见的状态码和意义

200 OK

请求正常处理完毕

204 No Content

请求成功处理，没有实体的主体返回

206 Partial Content

GET范围请求已成功处理

301 Moved Permanently

永久重定向，资源已永久分配新URI

302 Found

临时重定向，资源已临时分配新U RI

303 See Other

临时重定向，期望使用GET定向获取

304 Not Modified

发送的附带条件请求未满足

307 Temporary Redirect

临时重定向，POST不会变成GET

400 Bad Request

请求报文语法错误或参数错误

401 Unauthorized

需要通过HTTP认证，或认证失败

403 Forbidden

请求资源被拒绝

404 Not Found

无法找到请求资源（服务器无理由拒绝）

500 Internal Server Error

服务器故障或Web应用故障

503 Service Unavailable

服务器超负载或停机维护

#### 57、分别从前端、后端、数据库阐述web项目的性能优化

该题目网上有很多方法，我不想截图网上的长串文字，看的头疼，按我自己的理解说几点 前端优化：

1、减少http请求、例如制作精灵图

2、html和CSS放在页面上部，javascript放在页面下面，因为js加载比HTML和Css加载 慢，所以要优先加载html和css,以防页面显示不全，性能差，也影响用户体验差

后端优化：

1、缓存存储读写次数高，变化少的数据，比如网站首页的信息、商品的信息等。应用程 序读取数据时，一般是先从缓存中读取，如果读取不到或数据已失效，再访问磁盘数据 **库,**并将数据再次写入缓存。

2、异步方式，如果有耗时操作，可以采用异步，比如celery

3、代码优化，避免循环和判断次数太多，如果多个if else判断，优先判断最有可能先发生的情况

数据库优化：

1、如有条件，数据可以存放于redis，读取速度快

2、建立索引、外键等

#### 58、使用pop和del删除字典中的"name"字段，dic={"name":"zs","age":18)

![](https://files.mdnice.com/user/11419/7bcdeabe-7c55-42fa-a0c2-061312b462c8.png)

#### 59、MYSQL数据存储引擎

InnoDB:支持事务处理，支持外键，支持崩溃修复能力和并发控制。如果需要对事务的 完整性要求比较高(比如银行)，要求实现并发控制(比如售票)，那选InnoDB有很大的优势。如果需要频繁的更新、删除操作的数据库，也可以选择InnoDB,因为支持事 务的提交(commit)和回滚(rollback)。

MylSAM:插入数据快，空间和内存使用比较低。如果表主要是用于插入新记录和读出 记录，那么选择MylSAM能实现处理高效率。如果应用的完整性、并发性要求比较低， 也可以使用。

MEMORY:所有的数据都在内存中，数据的处理速度快，但是安全性不高。如果需要很 快的读写速度，对数据的安全性要求较低，可以选择MEMOEY.它对表的大小有要求， 不能建立太大的表。所以，这类数据库只使用在相对较小的数据库表。

#### 60、zip函数历史文章已经说了，得出［("a",1),("b",2), ("c",3).（"d",4）,("e",5)

![](https://files.mdnice.com/user/11419/b9e27698-2f62-49e4-baab-f7a7df711d0b.png)



dict（）创建字典新方法

![](https://files.mdnice.com/user/11419/a42ecf97-4a70-4af9-85f3-fc0f2b2e0049.png)

#### 61、简述同源策略

同源策略需要同时满足以下三点要求：

1）	协议相同

2）	域名相同

3）	端口相同

http:www.test.com与https:www.test.com 不同源	协议不同

http:www.test.com与http:www.admin.com 不同源	域名不同

http:www.test.com 与 http:[www.test.com:8081](http://www.test.com:8081) 不同源	端口不同

只要不满足其中任意一个要求，就不符合同源策略，就会出现"跨域"

#### 62、简述cookie和session的区别

1、 session在服务器端，cookie在客户端（浏览器）

2、session的运行依赖session id,而session id是存在cookie中的，也就是说，如 果浏览器禁用了 cookie ,同时session也会失效，存储Session时，键与Cookie中的 sessionid相同，值是开发人员设置的键值对信息，进行了base64编码，过期时间由开发 人员设置

3、cookie安全性比session差

#### 63、简述多线程、多进程

进程：

1、操作系统进行资源分配和调度的基本单位，多个进程之间相互独立

2、稳定性好，如果一个进程崩溃，不影响其他进程，但是进程消耗资源大，开启的进程数量有限制

线程：

1、CPU进行资源分配和调度的基本单位，线程是进程的一部分，是比进程更小的能独立 运行的基本单位，一个进程下的多个线程可以共享该进程的所有资源

2、如果I0操作密集，则可以多线程运行效率高，缺点是如果一个线程崩溃，都会造成进程的崩溃

应用：

I0密集的用多线程，在用户输入，sleep时候，可以切换到其他线程执行，减少等待的时间

CPU密集的用多进程，因为假如I0操作少，用多线程的话，因为线程共享一个全局解释 器锁，当前运行的线程会霸占GIL,其他线程没有GIL,就不能充分利用多核CPU的优势

#### 64、简述any()和all()方法

any():只要迭代器中有一个元素为真就为真

all():迭代器中所有的判断项返回都是真，结果才为真

python中什么元素为假？

答案：(0,空字符串，空列表、空字典、空元组、None, False)

![](https://files.mdnice.com/user/11419/4b39440c-ea03-4529-8a65-05e44c46c63d.png)

测试all()和any ()方法

![](https://files.mdnice.com/user/11419/5e4b6a78-bae1-4619-ab39-a4267d80f3fd.png)



#### 65、lOError、 AttributeError、ImportError、 IndentationError、IndexEiror、KeyError、SyntaxError、NameError分别代表什么异常

lOError:输入输出异常

AttributeError:试图访问一个对象没有的属性

ImportError:无法引入模块或包，基本是路径问题

IndentationError:语法错误，代码没有正确的对齐

IndexError:下标索引超出序列边界

KeyError:试图访问你字典里不存在的键

SyntaxError:Python代码逻辑语法出错，不能执行

NameError:使用一个还未赋予对象的变量

#### 66、python中copy和deepcopy区别

1、复制不可变数据类型，不管copy还是deepcopy渚5是同一个地址当浅复制的值是不可 变对象（数值，字符串，元组）时和=”赋值”的情况一样，对象的id值与浅复制原来的值相同。

![](https://files.mdnice.com/user/11419/012f3cf1-a957-4c39-9bde-e43bb89b1375.png)

2、复制的值是可变对象（列表和字典）

浅拷贝copy有两种情况:

第一种情况：复制的对象中无复杂子对象，原来值的改变并不会影响浅复制的值，同时浅复制的值改变也并不会影响原来的值。原来值的id值与浅复制原来的值不同。

第二种情况：复制的对象中有复杂子对象（例如列表中的一个子元素是一个列 表）,改变原来的值中的复杂子对象的值,会影响浅复制的值。

深拷贝deepcopy:完全复制独立，包括内层列表和字典

![](https://files.mdnice.com/user/11419/130ca296-0843-498f-bd20-71f028c9b1fb.png)

![](https://files.mdnice.com/user/11419/53fdd569-9669-459e-81ad-608ddfa6a18f.png)

#### 67、列出几种魔法方法并简要介绍用途

__init__:对象初始化方法

__new__:创建对象时候执行的方法，单列模式会用到

__str__:当使用print输出对象的时候，只要自己定义了_str_（self）方法，那么就会打印 从在这个方法中return的数据

__del__:删除对象执行的方法

#### 68、C:\Users\ry-wu.junya\Desktop>python 1.py22 33命令行启动程序并传参，\print（sys.argv）会输出什么数据？

文件名和参数构成的列表

![](https://files.mdnice.com/user/11419/3bd1c242-b3d4-4788-b1ed-df03c02d0f1f.png)

iC: \Users\ry-wu. junya\Desktcip>python 1. py 22 33 Dl.py, *22J, '33J]

#### 69、请将[i for i in range（3）]改成生成器

生成器是特殊的迭代器，

1、	列表表达式的【】改为0即可变成生成器

2、	函数在返回值得时候出现yield就变成生成器，而不是函数了; 中括号换成小括号即可

![](https://files.mdnice.com/user/11419/efc0df1c-319c-494f-a6ca-87e675138d9c.png)

#### 70、a = '' hehheh"，去除收尾空格

![](https://files.mdnice.com/user/11419/617740b1-6c53-409e-ba68-11394e675339.png)

#### 71、举例sort和sorted对列表排序，list=[0,-1,3，-10,5，9]

![](https://files.mdnice.com/user/11419/f123902d-d187-4165-b34a-c5adb69fcefe.png)

#### 72、对list排序foo=[-5,8,0,4,9,-4,-20,-2,8,2,-4]，使用lambda函数从小到大排序

![](https://files.mdnice.com/user/11419/239c1b2c-784f-463a-881e-297270996f62.png)

#### 73、使用lambda函数对list排序foo = [-5,8,0,4,9,-4,-20,-2,8,2.-4],输出结果为[0,2,4,8,9,-2,-4,-4,-5,-20],正数从小到大，负数从大到小

(传两个条件，x<0?[|abs(x))

![](https://files.mdnice.com/user/11419/8ac39616-1495-4522-9cf2-90db8c634579.png)

#### 74、列表嵌套字典的排扇分别根据年龄和姓名排序

foo = [(nnamen:"zs"fwagen:19},(nname":nll"/,,age,,:54},

 (nname,,:nwa"/nage,,:17}/{,,namen:',dfnfnagen:23}]

​                 ![](https://files.mdnice.com/user/11419/280cade2-54fa-49a5-9eeb-f74bf3eb3032.png) 

#### 75、列表嵌套元组，分别按字母和数字排序

![](https://files.mdnice.com/user/11419/842a24d1-9587-4f40-b2b0-8a63cc137b83.png)

#### 76、列表嵌套列表排序，年龄数字相同怎么办?

![](https://files.mdnice.com/user/11419/e87a20e7-188b-4cbd-bc81-857518366ba2.png)

#### 77、根据键对字典排序(方法一，zip函数）

![](https://files.mdnice.com/user/11419/cd319120-63b8-4083-810a-73ef8f834686.png)



#### 78、根据键对字典排序(方法二不用zip)

有没有发现dic,items和zip(dic,keys(),dic.values())都是为了构造列表嵌套字典的结构, 方便后面用sorted。构造排序规则

![](https://files.mdnice.com/user/11419/780997f2-5c54-42c5-918f-3b7fa4a4dbbf.png)

#### 79、列表推导式、字典推导式、生成器

![](https://files.mdnice.com/user/11419/2fb19465-b913-4092-ada0-5e6c6bdb7ddb.png)

#### 80、出一道检验题目，根据字符串长度，看排序是否灵活运用

![](https://files.mdnice.com/user/11419/3387b23a-07b6-4c25-befe-a2023a6f8231.png)

#### 81、举例说明SQL注入和解决办法

当以字符串格式化书写方式的时候，如果用户输入的有;+SQL语句，后面的SQL语句会执行，比如例子中的SQL注入会删除数据库demo

​               ![](https://files.mdnice.com/user/11419/0053cd43-ef39-4510-a5bc-c0be642d5a06.png) 

解决方式：通过传参数方式解决SQL注入

![](https://files.mdnice.com/user/11419/44887096-2b64-42bc-adc8-1d593d912156.png)

#### 82、s="info:xiaoZhang 33 shandong",用正则切分字符串输出['info', 'xiaoZhang‘, '33', 'shandong']

I表示或，根据冒号或者空格切分

![](https://files.mdnice.com/user/11419/9344cf3a-9846-4d6c-962f-bc8a1632d124.png)

#### 83、正则匹配以163.com结尾的邮箱

![](https://files.mdnice.com/user/11419/c1deab70-6a95-4236-8b65-f77d400b8034.png)

#### 84、递归求和

![](https://files.mdnice.com/user/11419/47e5c0ef-1406-4d83-8135-2c99b2af2b08.png)

#### 85、python字典和json字符串相互转化方法

json.dumps()字典转json字符串，json.loads()json转字典

​                          ![](https://files.mdnice.com/user/11419/4f90ec6e-3872-42be-94d6-7b028c43798d.png)



#### 86、MyISAM与InnoDB 区别：

1、InnoDB支持事务，MylSAM不支持，这一点是非常之重要。事务是一种高 级的处理方式，如在一些列增删改中只要哪个出错还可以回滚还原，而MylSAM 就不可以了；

2、MylSAM适合査询以及插入为主的应用，InnoDB适合频繁修改以及涉及到 安全性较高的应用；

3、InnoDB支持外键，MylSAM不支持；

4、对于自增长的字段，InnoDB中必须包含只有该字段的索引，但是在MylSAM 表中可以和其他字段一起建立联合索引；

5、清空整个表时，InnoDB是一行一行的删除，效率非常慢。MylSAM则会重 建表；

#### 87、统计字符串中某字符出现次数

![](https://files.mdnice.com/user/11419/01b4e4fb-22b4-48c7-bb11-d41e6ad9f9ed.png)

#### 88、字符串转化大小写

![](https://files.mdnice.com/user/11419/7b40897b-8f4c-48b4-ad6f-3e13184e0b30.png)

#### 89、用两种方法去空格

![](https://files.mdnice.com/user/11419/ee49360e-4023-49fe-96ba-845a95c9f485.png)

#### 90、 正则匹配不是以4和7结尾的手机号

![](https://files.mdnice.com/user/11419/3e8ed52e-f8af-48ae-a42f-ae1c7fb6fe9a.png)

#### 91、简述python引用计数机制

python垃圾回收主要以引用计数为主，标记-清除和分代清除为辅的机制，其中标记-清除分代回收主要是为了处理循环引用的难题。

引用计數算法

当有1个变量保存了对象的引用时，此对象的引用计数就会加1

当使用del删除变量指向的对象时，如果对象的引用计数不为1,比如3,那么此时只会让 这个引用计数减1,即变为2,当再次调用del时，变为1,如果再调用1次del,此时会真的把对象进行删除

![](https://files.mdnice.com/user/11419/6295e1a4-3c61-418e-bedc-8c61ec19076b.png)

#### 92、int("1.4"），int(1.4)输出结果?

int(”1.4”)报错，int(1.4)输出 1

#### 93、列举3条以上PEP8编码规范

1、顶级定义之间空两行，比如函数或者类定义。

2、方法定义、类定义与第一个方法之间，都应该空一行

3、三引号进行注释

4、使用Pycharm、Eclipse—般使用4个空格来缩进代码

#### 94、正则表达式匹配第一个URL

![](https://files.mdnice.com/user/11419/fc5ba29e-3cc7-4c56-96e3-ea7e7be1a9f5.png)

#### 95、正则匹配中文

![](https://files.mdnice.com/user/11419/ddcdc500-4715-47a9-ad6f-425265ea569c.png)

#### 96、简述乐观锁和悲观锁

悲观锁，就是很悲观，每次去拿数据的时候都认为别人会修改，所以每次在拿数据的时候都会上锁，这样别人想拿这个数据就会blockM到它拿到锁。传统的关系型数据库里边就 用到了很多这种锁机制，比如行锁，表锁等，读锁，写锁等，都是在做操作之前先上锁。

乐观锁，就是很乐观，每次去拿数据的时候都认为别人不会修改，所以不会上锁，但是在更新的时候会判断一下在此期间别人有没有去更新这个数据，可以使用版本号等机制，乐 观锁适用于多读的应用类型，这样可以提高吞吐量

#### 97、r、r+、rb、rb+文件打开模式区别

模式较多，比较下背背记记即可

![](https://files.mdnice.com/user/11419/b7a24a73-754c-4a58-9b28-e97c10aec86e.png)

访问模 式   说明

r，以只读方式打开文件，文件的指针将会放在文件的幵头，这是默认模式

w，打幵一个文件只再于写入，如果该文件己存在则将其覆盖。如果该文件不存在，创建新文件

a，打开一个文件用于追加，如果该文件已存在，文件指针将会放在文件的结尾，也就是说，新的内容将会被写入到已有内容之后，如果该文件不存在，创建新文件进行写入

rb，以二選制格式打幵一个文件用于只读，文件指针将会放在文件的开头，这是默认模式

wb，以二进制格式打开一个文件只用于写入，如果该文件已存在如将其覆盖 ，如果该文件不存在， 建新文件。

ab，以二进制格式打开一个文件用于追加，如果该文件已存在，文件指针将会放在文件的结尾，也就 是说，新的内容将会被写入到已有内容之后。如果该文件不存在、创建新文件击行写入。

r+，打开一个文件用于读写，文件指针将会放在文件的开头。

W+，打开一个文件用于读写。如果该文件已存在则将其覆盖*虹果该文件不存在，创建新文件。

a+，打开一个文件用于读写，如果该文件已存在，文件指针将会放在文件的结尾，文件打幵时会是追加模式，如果该文件不存在，创建新文件用于读写

rb+，以二进制格式打幵一个文件用于读写，文件指针将会放在文件的开头。

wb+，以二进制格式打开一个文件用于读写，如果该文件已存在则将其覆盖，如果该文件不存在，创建新文件。

ab+，以二送制格式打开一个文件.用于追加，如果该文件已存在，文件指针将会放在文件的结尾。如果该文件不存在，创建新文件用于读写。

#### 98、Linux命令重定向 > 和>>

Linux允许将命令执行结果重定向到一个文件 将本应显示在终端上的内容输出/追加到指定文件中

\>表示输出，会覆盖文件原有的内容

\>>表示追加，会将内容追加到已有文件的末尾

用法示例:

将echo输出的信息保存到l.txt里echo Hello Python > l.txt

将tree输出的信息追加到l.txt文件的末尾tree >> l.txt

#### 99、正则表达式匹配出html><h1>www.itcast.cn</h1 ></html>

前面的 <> 和后面的 <> 是对应的，可以用此方法

​                                                     ![](https://files.mdnice.com/user/11419/b4ae2fcb-71b1-4fee-9e74-a37090778749.png)

#### 100、python传参数是传值还是传址？

Python中函数参数是引用传递（注意不是值传递）。对于不可变类型（数值型、字符 串、元组），因变量不能修改，所以运算不会影响到变量自身；而对于可变类型（列表字典）来说，函数体运算可能会更改传入的参数变量。

![](https://files.mdnice.com/user/11419/eefa81be-2cdf-4c58-ae16-a44f3dc4abe9.png)

#### 101、求两个列表的交集,差集,并集

![](https://files.mdnice.com/user/11419/d833c9be-29cc-4309-adfd-92ae7847f0bf.png)



#### 102、生成0-100的随机数

![](https://files.mdnice.com/user/11419/849f12ec-ae20-43bc-8e86-0fc517f4a7a9.png)



#### 103、lambda匿名函散好处

精简代码，lambda省去了定义函数，map省去了写for循环过程

![](https://files.mdnice.com/user/11419/dbd6c8f8-51ca-4116-a95a-ee4ba881ed48.png)

#### 104、常见的网络传输协议

UDP、TCP、FTP、HTTP、SMTP等等

#### 105、单引号、双引号，三引号用法

1、单引号和双引号没有什么区别，不过单引号不用按shift,打字稍微快一点.表示字符 串的时候，单引号里面可以用双引号，而不用转义字符，反之亦然。

'She said:"Yes." ' or "She said: 'Yes.'"

2、但是如果直接用单引号扩住单引号，则需要转义，像这样：

’ She said:\'Yes.\''

3、	三引号可以直接书写多行，通常用于大段，大篇幅的字符串

"""

hello

world

"""

#### 106、python垃圾回收机制

python垃圾回收主要以引用计数为主，标记-清除和分代清除为辅的机制，其中标记分代回收主要是为了处理循环引用的难题。

***\*引用计数算法\****

当有1个变量保存了对象的引用时，此对象的引用计数就会加1

当使用del删除变量指向的对象时，如果对象的引用计数不为1,比如3,那么此时只会让这个引用计数减1,即变为2,当再次调用del时，变为1,如果再调用1次del,此时会真的把对象进行删除

![](https://files.mdnice.com/user/11419/53f11bf1-fa9a-4121-9803-a89eda01f1b4.png)



#### 107、HTTP请求中get和post区别

1、GET请求是通过URL直接请求数据，数据信息可以在URL中直接看到，比如浏览器访问；而POST请求是放在请求头中的，我们是无法直接看到的；

2、GET提交有数据大小的限制，一般是不超过1024个字节，而这种说法也不完全准确， HTTP协议并没有设定URL字节长度的上限，而是浏览器做了些处理，所以长度依据浏览 器的不同有所不同；POST请求在HTTP协议中也没有做说明，一般来说是没有设置限制 的，但是实际上浏览器也有默认值。总体来说，少量的数据使用GET,大量的数据使用 POST。

3、GET请求因为数据参数是暴露在URL中的，所以安全性比较低，比如密码是不能暴露 的，就不能使用GET请求；POST请求中，请求参数信息是放在请求头的，所以安全性较 高，可以使用。在实际中，涉及到登录操作的时候，尽量使用HTTPS请求，安全性更好

#### 108、python中读取Excel文件的方法

应用数据分析库pandas

![](https://files.mdnice.com/user/11419/93cb8e93-9c64-410d-be3a-c5b2e5b02775.png)

#### 109、简述多线程,多进程

进程：

1、操作系统进行资源分配和调度的基本单位，多个进程之间相互独立

2、稳定性好，如果一个进程崩溃，不影响其他进程，但是进程消耗资源大，开启的进程有限制

线程：

1、CPU进行资源分配和调度的基本单位，线程是进程的一部分，是比进程更小的能独立 运行的基本单位，一个进程下的多个线程可以共享该进程的所有资源

2、如果10操作密集，则可以多线程运行效率高，缺点是如果一个线程崩溃，都会造成进程的崩溃

应用：

IO密集的用多线程，在用户输入，Sleep时候，可以切换到其他线程执行，减少等待的时间

CPU密集的用多进程，因为假如10操作少，用多线程的话，因为线程共享一个全局解释 器锁，当前运行的线程会霸占GIL,其他线程没有GIL,就不能充分利用多核CPU的优势

#### 110、python正则中search和match

![](https://files.mdnice.com/user/11419/29271269-527a-4633-af42-6bfe9d90ea90.png)