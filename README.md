
# Object Oriented Programming

## Agenda
2. Describe what a class is in relation to Object Oriented Programming
3. Write a class definition, instantiate an object, define/inspect parameters, define/call class methods, define/code __init__ 
4. Overview of Inheritance
5. Important data science tools through the lens of objects: Standard Scaler and one-hot-encoder

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

## 5. Important data science tools through the lens of objects: 

We are becomming more and more familiar with a series of methods with names such as fit or fit_transform.

After instantiating an instance of a Standard Scaler, Linear Regression model, or One Hot Encoder, we use fit to learn about the dataset and save what is learned. What is learned is saved in the attributes.

### 1. Standard Scaler 

The standard scaler takes a series and, for each element, computes the absolute value of the difference from the point to the mean of the series, and divides by the standard deviation.

$\Large z = \frac{|x - \mu|}{s}$

What attributes and methods are available for a Standard Scaler object? Let's check out the code on [GitHub](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/preprocessing/_data.py)!

## Attributes

### `.scale_`


```python
ss.transform([ss.mean_])
```




    array([[0.]])



## Exercise One-hot Encoder

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


