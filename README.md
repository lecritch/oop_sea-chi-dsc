
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
  - You will develop knowledge of the OOP family of programming languages, what are the strengths and weakness of Python, and the strengths and weaknesses of other language families.

  
Let's begin by taking a look at the source code for [Sklearn's standard scalar](https://github.com/scikit-learn/scikit-learn/blob/fd237278e/sklearn/preprocessing/_data.py#L517)

Take a minute to peruse the source code on your own.  



# 2. "Everything in Python is an object"  


Python is an object-oriented programming language. You'll hear people say that "everything is an object" in Python. What does this mean?

Go back to the idea of a function for a moment. A function is a kind of abstraction whereby an algorithm is made repeatable. So instead of coding:

or even:

I can write:

Now imagine a further abstraction: Before, creating a function was about making a certain algorithm available to different inputs. Now I want to make that function available to different **objects**.

Even Python integers are objects. Consider:

We can see what type of object a variable is with the built-in type operator:

By setting x equal to an integer, I'm imbuing x with the methods of the integer class.

Python is dynamically typed, meaning you don't have to instruct it as to what type of object your variable is.  
A variable is a pointer to where an object is stored in memory.

For more details on this general feature of Python, see [here](https://jakevdp.github.io/WhirlwindTourOfPython/03-semantics-variables.html).
For more on shallow and deepcopy, go [here](https://docs.python.org/3/library/copy.html#copy.deepcopy)

# 3. Define attributes, methods, and dot notation

Dot notation is used to access both attributes and methods.

Take for example our familiar friend, the [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)

Instance attributes are associated with each unique object.
They describe characteristics of the object, and are accessed with dot notation like so:

What are some other DataFrame attributes we know?:


```python
# Other attributes
print(df.columns)
print(df.index)
print(df.dtypes)
print(df.T)
```

    Index(['price', 'sqft'], dtype='object')
    RangeIndex(start=0, stop=3, step=1)
    price    int64
    sqft     int64
    dtype: object
              0    1    2
    price    50   40   30
    sqft   1000  950  500


A **method** is what we call a function attached to an object

What other DataFrame methods do we know?


```python
df.describe()
df.copy()
df.head()
df.tail()
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
      <td>50</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>40</td>
      <td>950</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>500</td>
    </tr>
  </tbody>
</table>
</div>



# Pair Exercise

Let's practice accessing the methods associated with the built in string class.  
You are given a string below: 

Your task is to fix is so it reads `Hello, World!` using string methods.  To practice chaining methods, try to do it in one line.
Use the [documentation](https://docs.python.org/3/library/stdtypes.html#string-methods), and use the inspect library to see the names of methods.


```python
example.swapcase().replace('0','o').strip().replace('?','!')
```




    'Hello, World!'



# 4. Describe the relationship of classes and objectes, and learn to code classes

Each object is an instance of a **class** that defines a bundle of attributes and functions (now, as proprietary to the object type, called *methods*), the point being that **every object of that class will automatically have those proprietary attributes and methods**.

A class is like a blueprint that describes how to create a specific type of object.

![blueprint](img/blueprint.jpeg)


## Classes

We can define **new** classes of objects altogether by using the keyword `class`:

### Methods

We can also write functions that are associated with each class.  
As said above, a function associated with a class is called a method.

Wait a second, what's that `self` doing? 

## Magic Methods

It is common for a class to have magic methods. These are identifiable by the "dunder" (i.e. **d**ouble **under**score) prefixes and suffixes, such as `__init__()`. These methods will get called **automatically** as a result of a different call, as we'll see below.

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
    WHEELS = 4
     # default arguments included now in __init__
    def __init__(self, doors=4, fwd=False, driver_mood='peaceful', moving=False):
        
        self.doors = doors
        self.fwd = fwd
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

## 5. Overview of inheritance

We can also define classes in terms of *other* classes, in which cases the new classes **inherit** the attributes and methods from the classes in terms of which they're defined.

Suppose we decided we want to create an electric car class.

## 6. Important data science tools through the lens of objects: 

We are becomming more and more familiar with a series of methods with names such as fit or fit_transform.

After instantiating an instance of a Standard Scaler, Linear Regression model, or One Hot Encoder, we use fit to learn about the dataset and save what is learned. What is learned is saved in the attributes.

### 1. Standard Scaler 

The standard scaler takes a series and, for each element, computes the absolute value of the difference from the point to the mean of the series, and divides by the standard deviation.

$\Large z = \frac{|x - \mu|}{s}$

What attributes and methods are available for a Standard Scaler object? Let's check out the code on [GitHub](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/preprocessing/_data.py)!

## Attributes

### `.scale_`

When we fit the standard scaler, it studies the object passed to it, and saves what is learned in its instance attributes


```python
ss.transform([ss.mean_])
```




    array([[0.]])



## Pair Exercise: One-hot Encoder

Another object that you will use often is OneHotEncoder from sklearn. It is recommended over pd.get_dummies() because it can trained, with the learned informed stored in the attributes of the object.

Let's interact with an important parameters which we can pass when instantiating the OneHotEncoder object:` drop`.  

By dropping column, we avoid the [dummy variable trap](https://en.wikipedia.org/wiki/Dummy_variable_(statistics)).  

By passing `drop = True`, sklearn drops the first category it happens upon.  In this case, that is 'su'.  But what if we want to drop 'm'.  We can pass an array like object in as parameter to specify which column to drop.





```python
# Instantiate a OneHotEncoder object

ohe = OneHotEncoder(drop=['m'])
```


```python
ohe_matrix = ohe.fit_transform(df[['days']])
```

It is a sparse matrix, which is a matrix that is composed mostly of zeros


```python
ohe_columns = list(ohe.categories_[0])
ohe_columns.pop(int(ohe.drop_idx_))
oh_df.columns = ohe_columns
oh_df.head()
oh_df.columns = ohe_columns
oh_df.head()
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
      <th>f</th>
      <th>s</th>
      <th>su</th>
      <th>t</th>
      <th>th</th>
      <th>w</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Now, add the onehotencoded columns to the original df, and drop the days column

df = df.join(oh_df).drop('days', axis=1)
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
      <th>orders</th>
      <th>f</th>
      <th>s</th>
      <th>su</th>
      <th>t</th>
      <th>th</th>
      <th>w</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>758</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>105</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>562</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>132</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


