## 浅谈C++中指针和引用的区别



在C++中，我们常常使用到指针和引用，但对于它们的区别，很多C++的老手也容易混淆。

下面我们就来浅谈一下C++中指针和引用的区别，而这也是在面试时常会被问到的。



**看文章之前，别忘了关注我们，在我们这里，有你所需要的干货哦！**

------

### 1.基础概念

#### （1）指针

> 在C/C++语言中，指针一般被认为是指针变量，指针变量的内容存储的是其指向的对象的首地址，指向的对象可以是变量（指针变量也是变量），数组，函数等占据存储空间的实体。

##### ①创建指针变量

```C++
//格式：type *var-name;
int    *ip;    /* 一个整型的指针 */
double *dp;    /* 一个 double 型的指针 */
float  *fp;    /* 一个浮点型的指针 */
char   *ch;    /* 一个字符型的指针 */
```



#### （2）引用

> 引用变量是一个别名，也就是说，它是某个已存在变量的另一个名字。一旦把引用初始化为某个变量，就可以使用该引用名称或变量名称来指向变量。

##### ①创建并初始化

```c++
//格式：type &var-name = var2;
//（注：var2为之前已定义的目标变量名）
int i = 17;
int &r = i;
double &s = d;
```

##### ②引用的三个特点

- 不存在空引用。引用必须连接到一块合法的内存。
- 一旦引用被初始化为一个对象，就不能被指向到另一个对象。
- 引用必须在创建时被初始化。



------

### 2.指针和引用的区别

#### (1)性质上的区别

##### ①关于创建和初始化的不同

指针：任何时候均可被初始化，指针可以在任何时候指向到另一个对象，即指向其它的存储单元。

引用：创建时就需要初始化，一旦引用被初始化为一个对象，就不能被指向到另一个对象。



##### ②关于实质

指针：是一个变量，只不过这个变量存储的是一个地址，指向内存的一个存储单元。

引用：跟原来的变量实质上是同一个东西，只不过是原变量的一个别名而已。对引用的操作与对变量直接操作完全一样。

**例如：**

```C++
//指针
int a=1;		//定义一个整型变量a
int *p=&a;		//定义一个指针变量p，该指针变量指向a的存储单元，即p的值是a存储单元的地址。
```

```c++
//引用
int a=1;		//定义一个整型变量a
int &b=a;		//定义一个引用b指向变量a，a和b是同一个东西，在内存占有同一个存储单元。
```



##### ③关于空（NULL）的概念

指针：存在空指针。

引用：不存在空引用。

**（注：空指针的定义和使用）**

```c++
#include <iostream>

using namespace std;

int main ()
{
  int *ptr = NULL;

  cout << "ptr 的值是 " << ptr ;
 
  return 0;
}

//如需检查一个空指针，您可以使用 if 语句，如下所示：
if(ptr)     /* 如果 ptr 非空，则完成 */
if(!ptr)    /* 如果 ptr 为空，则完成 */
```

**【补充说明】**
***NULL的二义性：***
		**例如：**

```
		void func(int i) {}
		void func(int *p) {}
```

​	以上两个重载的func函数，很明显传入的参数是不同的。如果我们需要调用func(int  * )函数,下面两种调用方式均无法达到目的。
​			<u>func(0);            ✘</u> 
​			<u>func(NULL);     ✘</u>
​	原因很简单，0是整数，NULL实际上也是整数0。当然也不是没有办法，代码可以改成这样：
​			<u>func((int*)0);             ✔</u>
​			<u>func((int*)NULL);     ✔</u>
​	虽然函数调用可以成功，但终究是不大舒服。

***解决方式：***为了更好的解决这个问题，C++11引入nullptr的概念。

​	分析NULL的二义性问题，不难发现其实我们只是需要一个字面值来表示空指针，而不是借用数字0。因此C++11中定义了这个字面值：nullptr。有个它，再调用output函数是就清晰多了。



##### ④关于多级

指针：可以有多级，如`int *ip;    /* 一个整型的指针 */` ，`int **p;    /* 一个整型指针的指针 */` 。

引用：只能是一级，如`int a = 1;int &b = a;  /* 一个整型变量的引用 */`  是合法的，但~~`int &&a`~~是不合法的。



##### ⑤关于const

指针：可以有const指针。

引用：没有const引用。



##### ⑥关于sizeof

"sizeof指针"：得到的是指针本身的大小。

"sizeof引用"：得到的是所指向的变量(对象)的大小。



##### ⑦关于运算

指针和引用的自增(++)运算意义不一样。



#### (2)作为参数传递时的区别

##### ①指针作为参数进行传递：

 => 如果要想达到也同时修改的目的的话，就得使用引用了。

```c++
#include<iostream>
using namespace std;

void swap(int *a,int *b)
{
　　int temp=*a;
　　*a=*b;
　　*b=temp;
}

void test(int *p)
{
　　int a=1;
　　p=&a;
　　cout<<p<<" "<<*p<<endl;
}

int main(void)
{
    int a=1,b=2;
    swap(&a,&b);
    cout<<a<<" "<<b<<endl;
    
    int *p=NULL;
    test(p);
    if(p==NULL)
    cout<<"指针p为NULL"<<endl;
    system("pause");
    return 0;
}
```

```C++
//运行结果为：
2 1       
0x61fddc 1
指针p为NULL
```

 => swap函数中，使用指针作为参数，传递过来的是实参的地址（即&a和&b）。因此使用\*a实际上是取存储在实参内存单元里的数据，则对\*a的赋值即是对传入实参的赋值，因此可以实现对实参进行改变的目的。

 => test函数中，事实上传递的也是地址，只不过传递的是指针地址。也就是说将指针变量作为参数进行传递时，事实上是“值传递”的方式，C语言中实参变量和形参变量之间的数据传递是单向的“值传递”方式。指针变量做函数参数也要遵循这一规则。当把指针作为参数进行传递时，也是将实参的一个拷贝传递给形参，即上面程序main函数中的p和test函数中使用的p不是同一个变量，存储2个变量p的单元也不相同（只是2个p指向同一个存储单元）。所以在test函数中对p进行修改，并不会影响到main函数中的p的值。





##### ②将引用作为函数的参数进行传递。

在讲引用作为函数参数进行传递时，实质上传递的是实参本身，而不是实参的拷贝，对形参的修改就是对实参的修改。因此在用引用进行参数传递时，不仅节约时间，而且可以节约空间。

**例如：**

```c++
#include<iostream>
using namespace std;

void test(int *&p)
{
　　int a=1;
　　p=&a;
　　cout<<p<<" "<<*p<<endl;
}

int main(void)
{
    int *p=NULL;
    test(p);
    if(p!=NULL)
    cout<<"指针p不为NULL"<<endl;
    system("pause");
    return 0;
}
```

```
//运行结果为：
0x22ff44 1
指针p不为NULL
```

这足以说明用引用进行参数传递时，事实上传递的是实参本身，而不是拷贝。

所以在上述要达到**同时修改指针**的目的的话，就得使用引用了。





### 总结

对于C++/C语言来说，如何使用指针是必考题，而与指针相关的引用也常会被作为面试题之一。能对它们进行区分辨别是很重要的。







### 其它干货

- [算法岗，不会写简历？我把它拆开，手把手教你写！](http://mp.weixin.qq.com/s?__biz=MzkzNDIxMzE1NQ==&mid=2247485095&idx=1&sn=b3fa4c5e87d2c883e4234a512b03f925&chksm=c241e5ebf5366cfd0e1e878d6f81cc441c39da645f53f470547a6e1ca8fad20d3de16f3055bb&scene=21#wechat_redirect)
- [(算法从业人员必备！)Ubuntu办公环境搭建！](http://mp.weixin.qq.com/s?__biz=MzkzNDIxMzE1NQ==&mid=2247485184&idx=1&sn=cc9ac830e1fccceac03b1ec18c4cdc84&chksm=c241e44cf5366d5ac977c3f78b2b83148a6dba80ab8213c31ecc77582fe2eb2d2991bb76ecfc&scene=21#wechat_redirect)
- [“我能分清奥特曼们了，你能分清我的口红吗？”](http://mp.weixin.qq.com/s?__biz=MzkzNDIxMzE1NQ==&mid=2247485606&idx=1&sn=a54673568dda61af44ff3a707dd52927&chksm=c241ebeaf53662fc27913f4ce84252efd7d996e16a30828d52dcd840de0868f2ae8f911dda09&scene=21#wechat_redirect)
- [入门算法，看这个呀！(资料可下载)](http://mp.weixin.qq.com/s?__biz=MzkzNDIxMzE1NQ==&mid=2247485678&idx=1&sn=1f4c265a29bc78f3c3470cdf328a2d7b&chksm=c241eba2f53662b487a3a0a629d97b1e811552153728031c2b30614aeadd722cc83bf1d3d866&scene=21#wechat_redirect)
- [放弃大厂算法Offer，去银行做开发，现在...](http://mp.weixin.qq.com/s?__biz=MzkzNDIxMzE1NQ==&mid=2247485716&idx=1&sn=ca48d6fd590c9a76749c41c47e5f2da3&chksm=c241ea58f536634e7b19eab8b6f14953e068b8701623fd8c1f3deb6e1abd26503e7062bddcfd&scene=21#wechat_redirect)
- [超6k字长文，带你纵横谈薪市场（建议工程师收藏！)](http://mp.weixin.qq.com/s?__biz=MzkzNDIxMzE1NQ==&mid=2247485766&idx=1&sn=e8c91387c1f8cb5902b695e73018a609&chksm=c241ea0af536631c7c9f01eac9e596536f1c666a824b6ea80915189b773473dd9e54ef26d751&scene=21#wechat_redirect)



### 引用

- https://www.cnblogs.com/dolphin0520/archive/2011/04/03/2004869.html
- https://www.runoob.com/cplusplus/cpp-references.html
- https://baike.baidu.com/item/%E6%8C%87%E9%92%88/2878304?fr=aladdin
- https://blog.csdn.net/craftsman1970/article/details/79794900
