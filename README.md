## BaCO

The Bayesian Compiler Optimization framework (BaCO) is a flexible out-of-the-box compiler optimizer. It is a flexible tool that could be applied in a wide range of compiler optimization and hardware design settings.

## Installation
Clone the repo and pip install the requirements listed in requirements.txt.

## Use
There are two main ways to interact with BaCO: either by calling the optimize() routine in baco/baco.py if you have a python interface to your compiler or through the client-server functionality that interacts with your application through the terminal. In either case, a .json scenario file is required that sets up the optimization. This is where the input parameters are given as well as other run settings. Examples of scenario files can be found in tests/aux and the full template is found in baco/schema.json. 

# Running BaCO with a black-box function
To run it with a blackbox function, simply call the optimize() routine with a callable python-function and the name of the scenario file.

# Running BaCO client-server
In the client-server mode, the compiler framework calls BaCO on demand asking for recommended settings. 

The two parties communicate via a client (the third-party software) and server (BaCO) protocol defined by BaCO. The general idea of the protocol is that the client asks the server the configurations to run. So, there is the first design of experiments phase where samples are drawn and a second phase that is about the (Bayesian) optimization. 

To enable the Client-Server mode add this line to the json file:

```
“baco_mode”: {
       “mode”: “client-server”
   }
```

The client and server communicate following a csv-like protocol via the standard output. The client calls BaCO to start the optimization process. When called by the client, BaCO will reply by requesting a number of function evaluations and wait for a reply. As an example, BaCO will reply with:

```
Request 3
x1,x2
-10,12
1,6
-8,20
```

Note that BaCO starts the protocol by stating how many evaluations it is requesting, followed by the input parameters (x1 and x2 int this case) and a series of parameter values.

Upon receiving this message, the client must compute the function values at the requested points and reply with the input parameters and the function values:

```
x1,x2,value,Valid
-10,12,267,False
1,6,28,True
-8,20,463,False
```

This protocol continues for the number of iterations specified by the client in the scenario file and after all iterations are done, BaCO will save all explored points to a csv file and end its execution. 

## Code release
The code will be released with additional descriptions and utility features, and will be continuously updated, after the paper revision. 
The easiest way to run the code is to run tests/test_all.py from the BaCO root folder. To run the code from anywhere other than the BaCO root folder, first add the baco root folder to your PYTHONPATH, using 'export PYTHONPATH=$PYTHONPATH:{path to baco root folder}'.

