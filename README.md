
# Object Oriented Programming

## Agenda

2. Describe what a class is in relation to Object Oriented Programming
3. Write a class definition, instantiate an object, define/inspect parameters, define/call class methods, define/code __init__ 
4. Overview of Inheritance

## 2.  Describe what a class is in relation to Object Oriented Programming

Python is an object-oriented programming language. You'll hear people say that "everything is an object" in Python. What does this mean?

Go back to the idea of a function for a moment. A function is a kind of abstraction whereby an algorithm is made repeatable. So instead of coding:


```python
print(3**2 + 10)
print(4**2 + 10)
print(5**2 + 10)
```

    19
    26
    35


or even:


```python
for x in range(3, 6):
    print(x**2 + 10)
```

    19
    26
    35


I can write:


```python
def square_and_add_ten(x):
    return x**2 + 10
```

Now imagine a further abstraction: Before, creating a function was about making a certain algorithm available to different inputs. Now I want to make that function available to different **objects**.

An object is what we get out of this further abstraction. Each object is an instance of a **class** that defines a bundle of attributes and functions (now, as proprietary to the object type, called *methods*), the point being that **every object of that class will automatically have those proprietary attributes and methods**.

A class is like a blueprint that describes how to create a specific type of object.

![blueprint](img/blueprint.jpeg)


Even Python integers are objects. Consider:


```python
x = 3
```

We can see what type of object a variable is with the built-in type operator:


```python
type(x)
```




    int



By setting x equal to an integer, I'm imbuing x with the attributes and methods of the integer class.


```python
x.bit_length()
```




    2




```python
x.__float__()
```




    3.0



For more details on this general feature of Python, see [here](https://jakevdp.github.io/WhirlwindTourOfPython/03-semantics-variables.html).

# Exercise

## Look up a different type and find either a class or attribute that you did not know existed

There is a nice library, inspect, which can be used to look at the different attributes and methods associated with builtin objects.



```python
import inspect

example = 1
inspect.getmembers(example)
```




    [('__abs__', <method-wrapper '__abs__' of int object at 0x102d8a5a0>),
     ('__add__', <method-wrapper '__add__' of int object at 0x102d8a5a0>),
     ('__and__', <method-wrapper '__and__' of int object at 0x102d8a5a0>),
     ('__bool__', <method-wrapper '__bool__' of int object at 0x102d8a5a0>),
     ('__ceil__', <function int.__ceil__>),
     ('__class__', int),
     ('__delattr__', <method-wrapper '__delattr__' of int object at 0x102d8a5a0>),
     ('__dir__', <function int.__dir__()>),
     ('__divmod__', <method-wrapper '__divmod__' of int object at 0x102d8a5a0>),
     ('__doc__',
      "int([x]) -> integer\nint(x, base=10) -> integer\n\nConvert a number or string to an integer, or return 0 if no arguments\nare given.  If x is a number, return x.__int__().  For floating point\nnumbers, this truncates towards zero.\n\nIf x is not a number or if base is given, then x must be a string,\nbytes, or bytearray instance representing an integer literal in the\ngiven base.  The literal can be preceded by '+' or '-' and be surrounded\nby whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.\nBase 0 means to interpret the base from the string as an integer literal.\n>>> int('0b100', base=0)\n4"),
     ('__eq__', <method-wrapper '__eq__' of int object at 0x102d8a5a0>),
     ('__float__', <method-wrapper '__float__' of int object at 0x102d8a5a0>),
     ('__floor__', <function int.__floor__>),
     ('__floordiv__',
      <method-wrapper '__floordiv__' of int object at 0x102d8a5a0>),
     ('__format__', <function int.__format__(format_spec, /)>),
     ('__ge__', <method-wrapper '__ge__' of int object at 0x102d8a5a0>),
     ('__getattribute__',
      <method-wrapper '__getattribute__' of int object at 0x102d8a5a0>),
     ('__getnewargs__', <function int.__getnewargs__()>),
     ('__gt__', <method-wrapper '__gt__' of int object at 0x102d8a5a0>),
     ('__hash__', <method-wrapper '__hash__' of int object at 0x102d8a5a0>),
     ('__index__', <method-wrapper '__index__' of int object at 0x102d8a5a0>),
     ('__init__', <method-wrapper '__init__' of int object at 0x102d8a5a0>),
     ('__init_subclass__', <function int.__init_subclass__>),
     ('__int__', <method-wrapper '__int__' of int object at 0x102d8a5a0>),
     ('__invert__', <method-wrapper '__invert__' of int object at 0x102d8a5a0>),
     ('__le__', <method-wrapper '__le__' of int object at 0x102d8a5a0>),
     ('__lshift__', <method-wrapper '__lshift__' of int object at 0x102d8a5a0>),
     ('__lt__', <method-wrapper '__lt__' of int object at 0x102d8a5a0>),
     ('__mod__', <method-wrapper '__mod__' of int object at 0x102d8a5a0>),
     ('__mul__', <method-wrapper '__mul__' of int object at 0x102d8a5a0>),
     ('__ne__', <method-wrapper '__ne__' of int object at 0x102d8a5a0>),
     ('__neg__', <method-wrapper '__neg__' of int object at 0x102d8a5a0>),
     ('__new__', <function int.__new__(*args, **kwargs)>),
     ('__or__', <method-wrapper '__or__' of int object at 0x102d8a5a0>),
     ('__pos__', <method-wrapper '__pos__' of int object at 0x102d8a5a0>),
     ('__pow__', <method-wrapper '__pow__' of int object at 0x102d8a5a0>),
     ('__radd__', <method-wrapper '__radd__' of int object at 0x102d8a5a0>),
     ('__rand__', <method-wrapper '__rand__' of int object at 0x102d8a5a0>),
     ('__rdivmod__', <method-wrapper '__rdivmod__' of int object at 0x102d8a5a0>),
     ('__reduce__', <function int.__reduce__()>),
     ('__reduce_ex__', <function int.__reduce_ex__(protocol, /)>),
     ('__repr__', <method-wrapper '__repr__' of int object at 0x102d8a5a0>),
     ('__rfloordiv__',
      <method-wrapper '__rfloordiv__' of int object at 0x102d8a5a0>),
     ('__rlshift__', <method-wrapper '__rlshift__' of int object at 0x102d8a5a0>),
     ('__rmod__', <method-wrapper '__rmod__' of int object at 0x102d8a5a0>),
     ('__rmul__', <method-wrapper '__rmul__' of int object at 0x102d8a5a0>),
     ('__ror__', <method-wrapper '__ror__' of int object at 0x102d8a5a0>),
     ('__round__', <function int.__round__>),
     ('__rpow__', <method-wrapper '__rpow__' of int object at 0x102d8a5a0>),
     ('__rrshift__', <method-wrapper '__rrshift__' of int object at 0x102d8a5a0>),
     ('__rshift__', <method-wrapper '__rshift__' of int object at 0x102d8a5a0>),
     ('__rsub__', <method-wrapper '__rsub__' of int object at 0x102d8a5a0>),
     ('__rtruediv__',
      <method-wrapper '__rtruediv__' of int object at 0x102d8a5a0>),
     ('__rxor__', <method-wrapper '__rxor__' of int object at 0x102d8a5a0>),
     ('__setattr__', <method-wrapper '__setattr__' of int object at 0x102d8a5a0>),
     ('__sizeof__', <function int.__sizeof__()>),
     ('__str__', <method-wrapper '__str__' of int object at 0x102d8a5a0>),
     ('__sub__', <method-wrapper '__sub__' of int object at 0x102d8a5a0>),
     ('__subclasshook__', <function int.__subclasshook__>),
     ('__truediv__', <method-wrapper '__truediv__' of int object at 0x102d8a5a0>),
     ('__trunc__', <function int.__trunc__>),
     ('__xor__', <method-wrapper '__xor__' of int object at 0x102d8a5a0>),
     ('bit_length', <function int.bit_length()>),
     ('conjugate', <function int.conjugate>),
     ('denominator', 1),
     ('from_bytes', <function int.from_bytes(bytes, byteorder, *, signed=False)>),
     ('imag', 0),
     ('numerator', 1),
     ('real', 1),
     ('to_bytes', <function int.to_bytes(length, byteorder, *, signed=False)>)]



Below, there are four different built in types. Each person will get a type.  
Use inspect to find methods or attributes that either you:
  - didn't know existsed
  - forgot existed
  - find especially useful


```python
import numpy as np

w = [1,2,3]
x = {1:1, 2:2}
y = 'A string'
z = 1.5

types = ['w', 'x', 'y', 'z']

mccalister = ['Adam', 'Amanda','Chum', 'Dann', 
 'Jacob', 'Jason', 'Johnhoy', 'Karim', 
'Leana','Luluva', 'Matt', 'Maximilian', ]

while len(mccalister) >= 3:
    new_choices = np.random.choice(mccalister, 3, replace=False)
    type_choice = np.random.choice(types, 1)
    types.remove(type_choice)
    print(new_choices, type_choice)
    for choice in new_choices:
        mccalister.remove(choice)

```

    ['Adam' 'Jason' 'Leana'] ['w']
    ['Karim' 'Amanda' 'Chum'] ['x']
    ['Matt' 'Johnhoy' 'Maximilian'] ['y']
    ['Jacob' 'Dann' 'Luluva'] ['z']


# 3. Write a class definition, instantiate an object, define/inspect parameters, define/call class methods 

## Classes

We can define **new** classes of objects altogether by using the keyword `class`:


```python
class Car:
    """Transportation object"""
    pass # This called a stub. It will allow us to create an empty class without and error
```


```python
# Instantiate a car object
ferrari =  Car()
type(ferrari)
```




    __main__.Car




```python
# We can give desceribe the ferrari as having four wheels

ferrari.wheels = 4
ferrari.wheels
```




    4




```python
# But wouldn't it be nice to not have to do that every time? 
# We assume the blueprint of a car will have include the 4 wheels specification
# and assign it as an attribute when building the class
```


```python
class Car:
    """Automotive object"""
    
    wheels = 4                      # These are attributes of *every* car.

```


```python
civic = Car()
civic.wheels
```




    4




```python
#  Then we can add more attributes
class Car:
    """Automotive object"""
    
    wheels = 4                      # These are attributes of *every* car.
    doors = 4

```


```python
ferrari = Car()
ferrari.doors
```




    4




```python
# But a ferrari does not have 4 doors! 
# These attributes can be overwritten 

ferrari.doors = 2
ferrari.doors
```




    2



### Methods

We can also write functions that are associated with each class.  
As said above, a function associated with a class is called a method.


```python
#  Then we can add more attributes
class Car:
    """Automotive object"""
    
    wheels = 4                      # These are attributes of *every* car.
    doors = 4

    def honk(self):                   # These are methods we can call on *any* car.
        print('Beep beep')
    
```


```python
ferrari = civic = Car()
ferrari.honk()
civic.honk()

```

    Beep beep
    Beep beep


Wait a second, what's that `self` doing? 

## Magic Methods

It is common for a class to have magic methods. These are identifiable by the "dunder" (i.e. **d**ouble **under**score) prefixes and suffixes, such as `__init__()`. These methods will get called **automatically**, as we'll see below.

For more on these "magic methods", see [here](https://www.geeksforgeeks.org/dunder-magic-methods-python/).

When we create an instance of a class, Python invokes the __init__ to initialize the object.  Let's add __init__ to our class.



```python
#  Then we can add more attributes
class Car:
    """Automotive object"""
    
    WHEELS = 4                      # Capital letters mean wheels is a constant
    
    def __init__(self, doors, sedan):
        
        self.doors = doors
        self.sedan = sedan
        

    def honk(self):                   # These are methods we can call on *any* car.
        print('Beep beep')
    
```

By adding doors and moving to init, we need to pass parameters when instantiating the object.


```python
civic = Car(4, True)
civic.doors
```




    4



We can also pass default arguments if there is a value for a certain parameter which is very common.


```python
#  Then we can add more attributes
class Car:
    """Automotive object"""
    
    WHEELS = 4                     
    
    # default arguments included now in __init__
    def __init__(self, doors=4, sedan=False):
        
        self.doors = doors
        self.sedan = sedan
        

    def honk(self):                  
        print('Beep beep')
    
```


```python
civic = Car(sedan=True)
```

#### Positional vs. Named arguments


```python
# we can pass our arguments without names
civic = Car(4, True)


```


```python
# or with names
civic = Car(doors=4, sedan=True)

```


```python
# or with a mix
civic = Car(4, sedan=True)

```


```python
# but only when positional precides named
civic = Car(doors = 4, True)
```


      File "<ipython-input-29-6046029021d3>", line 2
        civic = Car(doors = 4, True)
                              ^
    SyntaxError: positional argument follows keyword argument




```python
# The self argument allows our methods to update our attributes.

#  Then we can add more attributes
class Car:
    """Automotive object"""
    
    WHEELS = 4                     
    
    # default arguments included now in __init__
    def __init__(self, doors=4, sedan=False, driver_mood='peaceful'):
        
        self.doors = doors
        self.sedan = sedan
        self.driver_mood = driver_mood
        

    def honk(self):                  
        print('Beep beep')
        self.driver_mood = 'pissed'
    
```


```python
civic = Car()
print(civic.driver_mood)
civic.honk()
print(civic.driver_mood)
```

    peaceful
    Beep beep
    pissed


# Pair

 Let's bring our knowledge together, and in pairs, code out the following:

We have an attribute `moving` which indicates, with a boolean, whether the car is moving or not.  

Fill in the functions stop and go to change the attribute `moving` to reflect the car's present state of motion after the method is called.  Also, include a print statement that indicates the car has started moving or has stopped.

Make sure the method works by calling it, then printing the attribute.



```python
#  Then we can add more attributes
class Car:
    """Automotive object"""
    
    # default arguments included now in __init__
    def __init__(self, doors=4, sedan=False, driver_mood='peaceful'):
        
        self.doors = doors
        self.sedan = sedan
        self.driver_mood = driver_mood
        
    def honk(self):                   # These are methods we can call on *any* car.
        print('Beep beep')
        
    def go(self):
        pass
    
    def stop(self):
        pass
```


```python
# run this code to make sure your 
civic = Car()
print(civic.moving)

civic.go()
print(civic.moving)

civic.stop()
print(civic.moving)
```

    False
    Whoa, that's some acceleration!
    True
    Screeech!
    False


## 4. Overview of inheritance

We can also define classes in terms of *other* classes, in which cases the new classes **inherit** the attributes and methods from the classes in terms of which they're defined.

Suppose we decided we want to create an electric car class.


```python
#  Then we can add more attributes
class ElectricCar(Car):
    """Automotive object"""
    
    # default arguments included now in __init__
    def __init__(self, hybrid=False):
        super().__init__()
        self.hybrid = True 
```


```python
prius = ElectricCar()
prius.honk()
```

    Beep beep



```python
#  And we can overwrite methods and parent attributes
class ElectricCar(Car):
    """Automotive object"""
    
    # default arguments included now in __init__
    def __init__(self, hybrid=False):
        
        # Prius owners are calmer than the average car owner
        super().__init__(driver_mood='serene')
        
        self.hybrid = True
        
    # overwrite inheritd methods
    
    def go(self):
        
        print('Whirrrrrr')
        self.moving = True
```


```python
prius = ElectricCar()
print(prius.moving)
prius.go()
prius.moving
print(prius.driver_mood)
```

    False
    Whirrrrrr
    serene



```python

```
