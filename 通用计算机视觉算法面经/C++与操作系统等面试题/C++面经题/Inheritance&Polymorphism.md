## 浅谈C++中继承与多态的区别



众所周知，C++有三大特性和五大原则，这三大特性分别是：**封装、继承和多态**。然而继承和多态这两个特性是许多人容易混淆的。

今天我们就来聊聊C++中继承和多态的区别。



**看文章之前，别忘了关注我们，在我们这里，有你所需要的干货哦！**

------

### 1.基础概念

#### （1）继承

> **指可以让某个类型的对象，获得另一种类型对象属性的方法。实际上就是类与类之间可以共用代码，实现代码重用。**
>
> 继承支持按级分类的概念。它可以使用现有类的所有功能，并在无需重新编写原来类的情况下对这些功能进行扩展。  
>
> 继承的过程：从一般到特殊的过程。
>
> 实现继承的方式：可以通过 “继承”（Inheritance）和“组合”（Composition）来实现。



***与继承相关的类类型：***

​	**基类：**被继承的类，即"父类"或“超类”。

​	**派生类：**基于基类创建的新类，又称“子类”。派生类可以访问基类中所有的非私有成员。

```C++
//派生类创建格式：
	class derived-class : access-specifier base-class
	/*
		①access-specifier表示继承类型。我们几乎不使用 protected 或 private 继承，通常使用 public 			继承。若如果未使用访问修饰符 access-specifier，则默认为private。
		②base-class表示已有基类
	*/
```

![](https://files.mdnice.com/user/15198/44276200-ae47-4b1a-b731-3372295bd677.png)

**代码实现：**

```c++
// 基类
 class Animal {
   // eat() 函数
   // sleep() 函数
 };

//派生类
 class Dog : public Animal {
   // bark() 函数
 };
```



***继承几种类型：***继承类型是通过上面讲解的访问修饰符 access-specifier 来指定的。

​	**private：私有继承**

​		基类private成员 不能继承
​		基类public、protected成员,可以继承,在派生类中需要通过private访问

​	**protected:保护继承:**

​		基类private成员 不能继承
​		基类public成员,可以继承,在派生类中相当于是protected访问
​		基类protected成员,可以继承,在派生类中相当于是protected访问

​	**public:公有继承:**

​		基类private成员 不能继承
​		基类public成员,可以继承,在派生类中相当于是public访问
​		基类protected成员,可以继承,在派生类中相当于是protected访问

| 派生方式      | 基类的public成员  | 基类的protected成员 | 基类的private成员 | 派生方式引起的访问属性变化概括               |
| ------------- | ----------------- | ------------------- | ----------------- | -------------------------------------------- |
| private派生   | 变为private成员   | 变为private成员     | 不可见            | 基类的非私有成员都成为派生类的私有成员       |
| protected派生 | 变为protected成员 | 变为private成员     | 不可见            | 基类的非私有成员在派生类中的访问属性都降一级 |
| public派生    | 仍为public成员    | 仍为protected成员   | 不可见            | 基类的非私有成员在派生类中的访问属性保持不变 |



#### （2）多态

> **按字面的意思就是多种形态，指一个类实例的相同方法在不同情况下有不同表现形式。**
>
> 多态机制使内部结构不同的对象可以共享相同的外部接口。即子类可以重写父类的某个函数，从而为这个函数提供不同于父类的行为。一个父类的多个子类可以为同一个函数提供不同的实现，从而在父类这个公共的接口下，表现出多种行为。
>
> 多态的使用场景：当类之间存在层次结构，并且类之间是通过继承关联时。这意味着，虽然针对不同对象的具体操作不同，但通过一个公共的类，它们（那些操作）可以通过相同的方式予以调用。



在C++中，多态性的实现和联编（也称绑定）这一概念有关。主要分为**静态联编**和**动态联编**两种

* *静态联编支持的多态性*  称为**编译时多态性（静态多态性）**。在C++中，编译时多态性是通过函数重载和模板实现的。利用函数重载机制，在调用同名函数时，编译系统会根据实参的具体情况确定索要调用的是哪个函数。
* *动态联编所支持的多态性*  称为**运行时多态（动态多态）**。在C++中，运行时多态性是通过虚函数来实现的。

再举一个通俗易懂的例子：比如购买车票，普通人是全价票；学生是半价票等。


***多态实现的三个条件：***

​	①必须是**公有继承**
​	②必须是通过基类的**指针或引用** 指向派生类对象 访问派生类方法
​	③基类的方法必须是**虚函数**，且完成了虚函数的重写

**例如：**

```C++
//tips：虚函数指在类的成员函数前加**virtual**关键字。
#include <iostream> 
using namespace std;

// 基类
class Shape {
   protected:
      int width, height;
   public:
      Shape( int a=0, int b=0)
      {
         width = a;
         height = b;
      }
      virtual int area()
      {
         cout << "Parent class area :" <<endl;
         return 0;
      }
};

// 派生类
class Rectangle: public Shape{
   public:
      Rectangle( int a=0, int b=0):Shape(a, b) { }
      int area ()
      { 
         cout << "Rectangle class area :" <<endl;
         return (width * height); 
      }
};

// 程序的主函数
int main( )
{
   Shape *shape;
   Rectangle rec(10,7);
 
   // 存储矩形的地址
   shape = &rec;
   // 调用矩形的求面积函数 area
   shape->area();
   
   return 0;
}

```

**运行结果：**`Rectangle class area`

（Tips： shape中的area函数若没有用virtual定义，则无法实现调用派生类中area函数的目的）



### 2.继承和多态的区别

区别：

1.多态的实现要求必须是共有继承。

2.继承关系中，并不要求基类方法一定是虚函数。而多态时，要求基类方法必须是虚函数。

3.多态：子类重写父类的方法，使得子类具有不同的实现。且运行时，根据实际创建的对象动态决定使用哪个方法。



#### 总结

​		在面向对象过程中，通常我们会以多个对象共有的特性作为基类进行创建。然后利用**继承的特性**，对基类进行派生。

​		基类与派生类存在相同的方法，但是有不同的方法体。当调用这些方法体时就需要利用C++的**多态性质**，根据对象的特性有选择的对方法进行调用。即多态是在不同继承关系的类对象，去调用同一函数，产生了不同的行为。





### 其它干货

- [算法岗，不会写简历？我把它拆开，手把手教你写！](http://mp.weixin.qq.com/s?__biz=MzkzNDIxMzE1NQ==&mid=2247485095&idx=1&sn=b3fa4c5e87d2c883e4234a512b03f925&chksm=c241e5ebf5366cfd0e1e878d6f81cc441c39da645f53f470547a6e1ca8fad20d3de16f3055bb&scene=21#wechat_redirect)
- [(算法从业人员必备！)Ubuntu办公环境搭建！](http://mp.weixin.qq.com/s?__biz=MzkzNDIxMzE1NQ==&mid=2247485184&idx=1&sn=cc9ac830e1fccceac03b1ec18c4cdc84&chksm=c241e44cf5366d5ac977c3f78b2b83148a6dba80ab8213c31ecc77582fe2eb2d2991bb76ecfc&scene=21#wechat_redirect)
- [“我能分清奥特曼们了，你能分清我的口红吗？”](http://mp.weixin.qq.com/s?__biz=MzkzNDIxMzE1NQ==&mid=2247485606&idx=1&sn=a54673568dda61af44ff3a707dd52927&chksm=c241ebeaf53662fc27913f4ce84252efd7d996e16a30828d52dcd840de0868f2ae8f911dda09&scene=21#wechat_redirect)
- [入门算法，看这个呀！(资料可下载)](http://mp.weixin.qq.com/s?__biz=MzkzNDIxMzE1NQ==&mid=2247485678&idx=1&sn=1f4c265a29bc78f3c3470cdf328a2d7b&chksm=c241eba2f53662b487a3a0a629d97b1e811552153728031c2b30614aeadd722cc83bf1d3d866&scene=21#wechat_redirect)
- [放弃大厂算法Offer，去银行做开发，现在...](http://mp.weixin.qq.com/s?__biz=MzkzNDIxMzE1NQ==&mid=2247485716&idx=1&sn=ca48d6fd590c9a76749c41c47e5f2da3&chksm=c241ea58f536634e7b19eab8b6f14953e068b8701623fd8c1f3deb6e1abd26503e7062bddcfd&scene=21#wechat_redirect)
- [超6k字长文，带你纵横谈薪市场（建议工程师收藏！)](http://mp.weixin.qq.com/s?__biz=MzkzNDIxMzE1NQ==&mid=2247485766&idx=1&sn=e8c91387c1f8cb5902b695e73018a609&chksm=c241ea0af536631c7c9f01eac9e596536f1c666a824b6ea80915189b773473dd9e54ef26d751&scene=21#wechat_redirect)



### 引用

- https://blog.csdn.net/qq_37185716/article/details/75044620
- https://www.runoob.com/cplusplus/cpp-inheritance.html
- https://www.runoob.com/cplusplus/cpp-polymorphism.html
