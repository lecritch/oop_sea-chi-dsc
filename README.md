
# Object Oriented Programming

## Agenda
1. Why a data scientist should learn about OOP
2. "Everything in Python is an object"  
3. Define attributes, methods, and dot notation
4. Describe the relationship of classes and objectes, and learn to code classes
5. Overview of Inheritance
6. Important data science tools through the lens of objects: Standard Scaler and One-Hot-Encoder

# 1. Why a data scientist should learn about OOP

![hackerman](https://media.giphy.com/media/MM0Jrc8BHKx3y/giphy.gif)

  - By becoming familiar with the principles of OOP, you will increase your knowledge of what's possible.  Much of what you might think you need to code by hand is already built into the objects.
  - With a knowledge of classes and how objects store information, you will develop a better sense of when the learning in machine learning occurs in the code, and after that learning occurs, how to access the information gained.
  - You become comfortable reading other people's code, which will improve your own code.
  - Improving your code will separate you from other candidates on the job market and in interviews.
  - You will also develop knowledge of the OOP family of programming languages, what are the strengths and weakness of Python, and the strengths and weaknesses of other language families.

  
Let's begin by taking a look at the source code for [Sklearn's standard scalar](https://github.com/scikit-learn/scikit-learn/blob/fd237278e/sklearn/preprocessing/_data.py#L517)

Take a minute to peruse the source code on your own.  



# 2. "Everything in Python is an object"  


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

Even Python integers are objects. Consider:


```python
x = 3
```

We can see what type of object a variable is with the built-in type operator:


```python
type(x)
```




    int



By setting x equal to an integer, I'm imbuing x with the methods of the integer class.


```python
x.bit_length()
```




    2




```python
x.__float__()
```




    3.0



Python is dynamically typed, meaning you don't have to instruct it as to what type of object your variable is.  
A variable is a pointer to where an object is stored in memory.


```python
# interesting side note about how variables operate in Python
```


```python
print(hex(id(x)))
```

    0x1012d45e0



```python
y = 3
```


```python
print(hex(id(y)))
```

    0x1012d45e0



```python
# this can have implications 

x_list = [1,2,3,4]
y_list = x_list

x_list.pop()
print(x_list)
print(y_list)
```

    [1, 2, 3]
    [1, 2, 3]



```python
# when you use copy(), you create a shallow copy of the object
z_list = y_list.copy()
y_list.pop()
print(y_list)
print(z_list)
```

    [1, 2]
    [1, 2, 3]



```python
a_list = [[1,2,3], [4,5,6]]
b_list = a_list.copy()
a_list[0][0] ='z'
b_list
```




    [['z', 2, 3], [4, 5, 6]]




```python
import copy

#deepcopy is needed for mutable objects
a_list = [[1,2,3], [4,5,6]]
b_list = copy.deepcopy(a_list)
a_list[0][0] ='z'
b_list
```




    [[1, 2, 3], [4, 5, 6]]



For more details on this general feature of Python, see [here](https://jakevdp.github.io/WhirlwindTourOfPython/03-semantics-variables.html).
For more on shallow and deepcopy, go [here](https://docs.python.org/3/library/copy.html#copy.deepcopy)

# 3. Define attributes, methods, and dot notation

Dot notation is used to access both attributes and methods.

Take for example our familiar friend, the [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)


```python
import pandas as pd
# Dataframes are another type of object which we are familiar with.

df = pd.DataFrame({'price':[50,40,30],'sqft':[1000,950,500]})
```


```python
type(df)
```




    pandas.core.frame.DataFrame



Instance attributes are associated with each unique object.
They describe characteristics of the object, and are accessed with dot notation like so:


```python
df.shape
```




    (3, 2)



What are some other DataFrame attributes we know?:


```python
# answer
```

A **method** is what we call a function attached to an object


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3 entries, 0 to 2
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   price   3 non-null      int64
     1   sqft    3 non-null      int64
    dtypes: int64(2)
    memory usage: 176.0 bytes



```python
# isna() is a method that comes along with the DataFrame object
df.isna()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>sqft</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



What other DataFrame methods do we know?

# Pair Exercise

Let's practice accessing the methods associated with the built in string class.  
You are given a string below: 


```python
example = '   hELL0, w0RLD?   '
```

Your task is to fix is so it reads `Hello, World!` using string methods.  To practice chaining methods, try to do it in one line.
Use the [documentation](https://docs.python.org/3/library/stdtypes.html#string-methods), and use the inspect library to see the names of methods.


```python
import inspect
inspect.getmembers(example)
```




    [('__add__', <method-wrapper '__add__' of str object at 0x115830078>),
     ('__class__', str),
     ('__contains__',
      <method-wrapper '__contains__' of str object at 0x115830078>),
     ('__delattr__', <method-wrapper '__delattr__' of str object at 0x115830078>),
     ('__dir__', <function str.__dir__()>),
     ('__doc__',
      "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."),
     ('__eq__', <method-wrapper '__eq__' of str object at 0x115830078>),
     ('__format__', <function str.__format__(format_spec, /)>),
     ('__ge__', <method-wrapper '__ge__' of str object at 0x115830078>),
     ('__getattribute__',
      <method-wrapper '__getattribute__' of str object at 0x115830078>),
     ('__getitem__', <method-wrapper '__getitem__' of str object at 0x115830078>),
     ('__getnewargs__', <function str.__getnewargs__>),
     ('__gt__', <method-wrapper '__gt__' of str object at 0x115830078>),
     ('__hash__', <method-wrapper '__hash__' of str object at 0x115830078>),
     ('__init__', <method-wrapper '__init__' of str object at 0x115830078>),
     ('__init_subclass__', <function str.__init_subclass__>),
     ('__iter__', <method-wrapper '__iter__' of str object at 0x115830078>),
     ('__le__', <method-wrapper '__le__' of str object at 0x115830078>),
     ('__len__', <method-wrapper '__len__' of str object at 0x115830078>),
     ('__lt__', <method-wrapper '__lt__' of str object at 0x115830078>),
     ('__mod__', <method-wrapper '__mod__' of str object at 0x115830078>),
     ('__mul__', <method-wrapper '__mul__' of str object at 0x115830078>),
     ('__ne__', <method-wrapper '__ne__' of str object at 0x115830078>),
     ('__new__', <function str.__new__(*args, **kwargs)>),
     ('__reduce__', <function str.__reduce__()>),
     ('__reduce_ex__', <function str.__reduce_ex__(protocol, /)>),
     ('__repr__', <method-wrapper '__repr__' of str object at 0x115830078>),
     ('__rmod__', <method-wrapper '__rmod__' of str object at 0x115830078>),
     ('__rmul__', <method-wrapper '__rmul__' of str object at 0x115830078>),
     ('__setattr__', <method-wrapper '__setattr__' of str object at 0x115830078>),
     ('__sizeof__', <function str.__sizeof__()>),
     ('__str__', <method-wrapper '__str__' of str object at 0x115830078>),
     ('__subclasshook__', <function str.__subclasshook__>),
     ('capitalize', <function str.capitalize()>),
     ('casefold', <function str.casefold()>),
     ('center', <function str.center(width, fillchar=' ', /)>),
     ('count', <function str.count>),
     ('encode', <function str.encode(encoding='utf-8', errors='strict')>),
     ('endswith', <function str.endswith>),
     ('expandtabs', <function str.expandtabs(tabsize=8)>),
     ('find', <function str.find>),
     ('format', <function str.format>),
     ('format_map', <function str.format_map>),
     ('index', <function str.index>),
     ('isalnum', <function str.isalnum()>),
     ('isalpha', <function str.isalpha()>),
     ('isascii', <function str.isascii()>),
     ('isdecimal', <function str.isdecimal()>),
     ('isdigit', <function str.isdigit()>),
     ('isidentifier', <function str.isidentifier()>),
     ('islower', <function str.islower()>),
     ('isnumeric', <function str.isnumeric()>),
     ('isprintable', <function str.isprintable()>),
     ('isspace', <function str.isspace()>),
     ('istitle', <function str.istitle()>),
     ('isupper', <function str.isupper()>),
     ('join', <function str.join(iterable, /)>),
     ('ljust', <function str.ljust(width, fillchar=' ', /)>),
     ('lower', <function str.lower()>),
     ('lstrip', <function str.lstrip(chars=None, /)>),
     ('maketrans', <function str.maketrans(x, y=None, z=None, /)>),
     ('partition', <function str.partition(sep, /)>),
     ('replace', <function str.replace(old, new, count=-1, /)>),
     ('rfind', <function str.rfind>),
     ('rindex', <function str.rindex>),
     ('rjust', <function str.rjust(width, fillchar=' ', /)>),
     ('rpartition', <function str.rpartition(sep, /)>),
     ('rsplit', <function str.rsplit(sep=None, maxsplit=-1)>),
     ('rstrip', <function str.rstrip(chars=None, /)>),
     ('split', <function str.split(sep=None, maxsplit=-1)>),
     ('splitlines', <function str.splitlines(keepends=False)>),
     ('startswith', <function str.startswith>),
     ('strip', <function str.strip(chars=None, /)>),
     ('swapcase', <function str.swapcase()>),
     ('title', <function str.title()>),
     ('translate', <function str.translate(table, /)>),
     ('upper', <function str.upper()>),
     ('zfill', <function str.zfill(width, /)>)]




```python
# we can also use built in dir() method
dir(example)
```




    ['__add__',
     '__class__',
     '__contains__',
     '__delattr__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__getitem__',
     '__getnewargs__',
     '__gt__',
     '__hash__',
     '__init__',
     '__init_subclass__',
     '__iter__',
     '__le__',
     '__len__',
     '__lt__',
     '__mod__',
     '__mul__',
     '__ne__',
     '__new__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__rmod__',
     '__rmul__',
     '__setattr__',
     '__sizeof__',
     '__str__',
     '__subclasshook__',
     'capitalize',
     'casefold',
     'center',
     'count',
     'encode',
     'endswith',
     'expandtabs',
     'find',
     'format',
     'format_map',
     'index',
     'isalnum',
     'isalpha',
     'isascii',
     'isdecimal',
     'isdigit',
     'isidentifier',
     'islower',
     'isnumeric',
     'isprintable',
     'isspace',
     'istitle',
     'isupper',
     'join',
     'ljust',
     'lower',
     'lstrip',
     'maketrans',
     'partition',
     'replace',
     'rfind',
     'rindex',
     'rjust',
     'rpartition',
     'rsplit',
     'rstrip',
     'split',
     'splitlines',
     'startswith',
     'strip',
     'swapcase',
     'title',
     'translate',
     'upper',
     'zfill']




```python
# Your code here
```

# 4. Describe the relationship of classes and objectes, and learn to code classes

Each object is an instance of a **class** that defines a bundle of attributes and functions (now, as proprietary to the object type, called *methods*), the point being that **every object of that class will automatically have those proprietary attributes and methods**.

A class is like a blueprint that describes how to create a specific type of object.

![blueprint](img/blueprint.jpeg)


## Classes

We can define **new** classes of objects altogether by using the keyword `class`:


```python
class Car:
    """Automotive object"""
    pass # This called a stub. 
```


```python
# Instantiate a car object
ferrari =  Car()
type(ferrari)
```




    __main__.Car




```python
# We can give describe the ferrari as having four wheels

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

It is common for a class to have magic methods. These are identifiable by the "dunder" (i.e. **d**ouble **under**score) prefixes and suffixes, such as `__init__()`. These methods will get called **automatically** as a result of a different call, as we'll see below.

For more on these "magic methods", see [here](https://www.geeksforgeeks.org/dunder-magic-methods-python/).

When we create an instance of a class, Python invokes the __init__ to initialize the object.  Let's add __init__ to our class.



```python
#  Then we can add more attributes
class Car:
    """Automotive object"""
    
    WHEELS = 4                      # Capital letters mean wheels is a constant
    
    def __init__(self, doors, fwd):
        
        self.doors = doors
        self.fwd = fwd
        

    def honk(self):                   # These are methods we can call on *any* car.
        print('Beep beep')
    
```

By adding doors and moving to init, we need to pass parameters when instantiating the object.


```python
civic = Car(4, True)
print(civic.doors)
print(civic.fwd)
```

    4
    True


We can also pass default arguments if there is a value for a certain parameter which is very common.


```python
#  Then we can add more attributes
class Car:
    """Automotive object"""
    
    WHEELS = 4                     
    
    # default arguments included now in __init__
    def __init__(self, doors=4, fwd=False):
        
        self.doors = doors
        self.fwd = fwd
        

    def honk(self):                  
        print('Beep beep')
    
```


```python
civic = Car()
print(civic.doors)
print(civic.fwd)
```

    4
    False


#### Positional vs. Named arguments


```python
# we can pass our arguments without names
civic = Car(4, True)

```


```python
# or with names
civic = Car(doors=4, fwd=True)

```


```python
# or with a mix
civic = Car(4, fwd=True)

```


```python
# but only when positional precides named
civic = Car(doors = 4, True)
```


      File "<ipython-input-268-6046029021d3>", line 2
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
    def __init__(self, doors=4, fwd=False, driver_mood='peaceful'):
        
        self.doors = doors
        self.fwd = fwd
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
    def __init__(self, doors=4, fwd=False, driver_mood='peaceful'):
        
        self.doors = doors
        self.fwd = fwd
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


## 5. Overview of inheritance

We can also define classes in terms of *other* classes, in which cases the new classes **inherit** the attributes and methods from the classes in terms of which they're defined.

Suppose we decided we want to create an electric car class.


```python
#  Then we can add more attributes
class ElectricCar(Car):
    """Automotive object"""
    
    pass
```


```python
prius = ElectricCar()
prius.honk()
prius.WHEELS
```

    Beep beep





    4




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


## 6. Important data science tools through the lens of objects: 

We are becomming more and more familiar with a series of methods with names such as fit or fit_transform.

After instantiating an instance of a Standard Scaler, Linear Regression model, or One Hot Encoder, we use fit to learn about the dataset and save what is learned. What is learned is saved in the attributes.

### 1. Standard Scaler 

The standard scaler takes a series and, for each element, computes the absolute value of the difference from the point to the mean of the series, and divides by the standard deviation.

$\Large z = \frac{|x - \mu|}{s}$

What attributes and methods are available for a Standard Scaler object? Let's check out the code on [GitHub](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/preprocessing/_data.py)!

## Attributes

### `.scale_`


```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# instantiate a standard scaler object
ss = StandardScaler()

# We can instantiate as many scaler objects as we want
maxs_scaler = StandardScaler()
```


```python
# Let's create a dataframe with two series

series_1 = np.random.normal(3,1,1000)
print(series_1.mean())
print(series_1.std())
```

    2.9226012273753232
    0.9780089271255945


When we fit the standard scaler, it studies the object passed to it, and saves what is learned in its instance attributes


```python
ss.fit(series_1.reshape(-1,1))

# standard deviation is saved in the attribute scale_
ss.scale_
```




    array([0.97800893])




```python
# mean is saved into the attribut mean
ss.mean_
```




    array([2.92260123])




```python
# Knowledge Check

# What value should I put into the standard scaler to make the equality below return 0

ss.transform([])
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-304-66adfde57247> in <module>
          3 # What value should I put into the standard scaler to make the equality below return 0
          4 
    ----> 5 ss.transform([])
    

    ~/anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/_data.py in transform(self, X, copy)
        793         X = check_array(X, accept_sparse='csr', copy=copy,
        794                         estimator=self, dtype=FLOAT_DTYPES,
    --> 795                         force_all_finite='allow-nan')
        796 
        797         if sparse.issparse(X):


    ~/anaconda3/lib/python3.7/site-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, warn_on_dtype, estimator)
        554                     "Reshape your data either using array.reshape(-1, 1) if "
        555                     "your data has a single feature or array.reshape(1, -1) "
    --> 556                     "if it contains a single sample.".format(array))
        557 
        558         # in the future np.flexible dtypes will be handled like object dtypes


    ValueError: Expected 2D array, got 1D array instead:
    array=[].
    Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.



```python
# we can then use these attributes to transform objects
np.random.seed(42)
random_numbers = np.random.normal(3,1, 2)
random_numbers
```




    array([3.49671415, 2.8617357 ])




```python
ss.transform(random_numbers.reshape(-1,1))
```




    array([[ 0.58702217],
           [-0.06223412]])




```python
# We can also use a scaler on a DataFrame
series_1 = np.random.normal(3,1,1000)
series_2 = np.random.uniform(0,100, 1000)
df_2 = pd.DataFrame([series_1, series_2]).T
ss_df = StandardScaler()
ss_df.fit_transform(df_2)

```




    array([[ 0.63918361, -1.63325007],
           [ 1.53240185,  1.50265028],
           [-0.260668  , -1.56258467],
           ...,
           [ 0.56254398, -1.61544876],
           [ 1.40620165, -1.36827099],
           [ 0.92178475, -0.56807826]])




```python
ss_df.transform([[5, 50]])
```




    array([[ 2.01911307, -0.00948621]])



## Pair Exercise: One-hot Encoder

Another object that you will use often is OneHotEncoder from sklearn. It is recommended over pd.get_dummies() because it can trained, with the learned informed stored in the attributes of the object.


```python
from sklearn.preprocessing import OneHotEncoder
```


```python
np.random.seed(42)
# Let's create a dataframe that has days of the week and number of orders. 

days = np.random.choice(['m','t', 'w','th','f','s','su'], 1000)
orders = np.random.randint(0,1000,1000)

df = pd.DataFrame([days, orders]).T
df.columns = ['days', 'orders']
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>days</th>
      <th>orders</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>su</td>
      <td>758</td>
    </tr>
    <tr>
      <th>1</th>
      <td>th</td>
      <td>105</td>
    </tr>
    <tr>
      <th>2</th>
      <td>f</td>
      <td>562</td>
    </tr>
    <tr>
      <th>3</th>
      <td>su</td>
      <td>80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>w</td>
      <td>132</td>
    </tr>
  </tbody>
</table>
</div>



Let's interact with an important parameters which we can pass when instantiating the OneHotEncoder object:` drop`.  

By dropping column, we avoid the [dummy variable trap](https://en.wikipedia.org/wiki/Dummy_variable_(statistics)).  

By passing `drop = True`, sklearn drops the first category it happens upon.  In this case, that is 'su'.  But what if we want to drop 'm'.  We can pass an array like object in as parameter to specify which column to drop.





```python
# Instantiate the OHE object with a param that tells it to drop Monday
ohe = None
```


```python
# Now, fit_transform the days column of the dataframe

ohe_array = None
```


```python
# look at __dict__ and checkout drop_idx_
# did it do what you wanted it to do?
ohe.__dict__
```




    {'categories': 'auto',
     'sparse': True,
     'dtype': numpy.float64,
     'handle_unknown': 'error',
     'drop': array(['m'], dtype=object),
     'categories_': [array(['f', 'm', 's', 'su', 't', 'th', 'w'], dtype=object)],
     'drop_idx_': array([1])}




```python
# check out the categories_ attribute
ohe.categories_
```




    [array(['f', 'm', 's', 'su', 't', 'th', 'w'], dtype=object)]




```python
# Check out the object itself
ohe_matrix
```




    <1000x6 sparse matrix of type '<class 'numpy.float64'>'
    	with 844 stored elements in Compressed Sparse Row format>



It is a sparse matrix, which is a matrix that is composed mostly of zeros


```python
# We can convert it to an array like so
oh_df = pd.DataFrame.sparse.from_spmatrix(ohe_matrix)
```


```python
# Now, using the categories_ attribute, set the column names to the correct days of the week
# you can use drop_idx_ for this as well


```


```python
# Add the onehotencoded columns to the original df, and drop the days column

```


```python

```
