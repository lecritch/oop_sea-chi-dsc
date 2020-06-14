
# Object Oriented Programming

## Agenda

2. Describe what a class is in relation to Object Oriented Programming
3. Write a class definition, instantiate an object, define/inspect parameters, define/call class methods, define/code __init__ 
4. Overview of Inheritance

## 2.  Describe what a class is in relation to Object Oriented Programming

Python is an object-oriented programming language. You'll hear people say that "everything is an object" in Python. What does this mean?

Go back to the idea of a function for a moment. A function is a kind of abstraction whereby an algorithm is made repeatable. So instead of coding:

or even:

I can write:

Now imagine a further abstraction: Before, creating a function was about making a certain algorithm available to different inputs. Now I want to make that function available to different **objects**.

An object is what we get out of this further abstraction. Each object is an instance of a **class** that defines a bundle of attributes and functions (now, as proprietary to the object type, called *methods*), the point being that **every object of that class will automatically have those proprietary attributes and methods**.

A class is like a blueprint that describes how to create a specific type of object.

![blueprint](img/blueprint.jpeg)


Even Python integers are objects. Consider:

We can see what type of object a variable is with the built-in type operator:

By setting x equal to an integer, I'm imbuing x with the attributes and methods of the integer class.

For more details on this general feature of Python, see [here](https://jakevdp.github.io/WhirlwindTourOfPython/03-semantics-variables.html).

# Exercise

## Look up a different type and find either a class or attribute that you did not know existed

There is a nice library, inspect, which can be used to look at the different attributes and methods associated with builtin objects.


Below, there are four different built in types. Each person will get a type.  
Use inspect to find methods or attributes that either you:
  - didn't know existsed
  - forgot existed
  - find especially useful

# 3. Write a class definition, instantiate an object, define/inspect parameters, define/call class methods 

## Classes

We can define **new** classes of objects altogether by using the keyword `class`:

### Methods

We can also write functions that are associated with each class.  
As said above, a function associated with a class is called a method.

Wait a second, what's that `self` doing? 

## Magic Methods

It is common for a class to have magic methods. These are identifiable by the "dunder" (i.e. **d**ouble **under**score) prefixes and suffixes, such as `__init__()`. These methods will get called **automatically**, as we'll see below.

For more on these "magic methods", see [here](https://www.geeksforgeeks.org/dunder-magic-methods-python/).

When we create an instance of a class, Python invokes the __init__ to initialize the object.  Let's add __init__ to our class.


By adding doors and moving to init, we need to pass parameters when instantiating the object.

We can also pass default arguments if there is a value for a certain parameter which is very common.

#### Positional vs. Named arguments

# Pair

 Let's bring our knowledge together, and in pairs, code out the following:

We have an attribute `moving` which indicates, with a boolean, whether the car is moving or not.  

Fill in the functions stop and go to change the attribute `moving` to reflect the car's present state of motion after the method is called.  Also, include a print statement that indicates the car has started moving or has stopped.

Make sure the method works by calling it, then printing the attribute.



```python
class Car:
    """Automotive object"""
    
     # default arguments included now in __init__
    def __init__(self, doors=4, sedan=False, driver_mood='peaceful', moving=False):
        
        self.doors = doors
        self.sedan = sedan
        self.moving = moving
        self.driver_mood = driver_mood

    def honk(self):                   # These are methods we can call on *any* car.
        print('Beep beep')
        
    def go(self):
        self.moving = True
        print('Whoa, that\'s some acceleration!')
    
    def stop(self):
        self.moving = False
        print('Screeech!')
```

## 4. Overview of inheritance

We can also define classes in terms of *other* classes, in which cases the new classes **inherit** the attributes and methods from the classes in terms of which they're defined.

Suppose we decided we want to create an electric car class.
