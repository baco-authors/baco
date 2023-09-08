BaCO
The Bayesian Compiler Optimization framework (BaCO) is a flexible out-of-the-box compiler optimizer. The aim is to develop a flexible tool that could be applied in a wide range of compiler optimization and hardware design settings.

This repo is connected to the publication https://arxiv.org/abs/2212.11142 and aims to facilitating reconstructing those results. It comes in two versions, one older that was used for the experiments in the paper and one, which is functionally equivalent but where the code has been cleaned up further. The older version is located in the branch Original.

If you want to use the optimizer for your own application, we would recommend using the maintained version of the codebase which will instead be kept under the name Hypermapper (https://github.com/luinardi/hypermapper, branch:Hypermapper-v3), which however won't include the compiler baselines.

Installation
Before starting, the ytopt baseline requires a modified version of scikit-optimize. We hence, strongly recommend setting up a virtual environment before installing BaCO.

To install BaCO, run

git clone https://github.com/baco-authors/baco.git
cd baco
pip install --upgrade pip
pip install -e .
To be able to run the compiler baselines you need to install ytopt and Opentuner. This is done by running

sh install_baselines.sh
To test the installation, run python tests/test_all.py in the tests folder.

There is a clash between the versions of scikit-learn used by ytopt (1.0.2) and the random forest implementation used by BaCO (1.3.0). Following the above instructions, it will have the ytopt-compatible installed.

Use
There are two main ways to interact with BaCO: either by calling the optimize() routine in baco/run.py if you have a python interface to your compiler or through the client-server functionality that interacts with your application through the terminal. In either case, a .json scenario file is required that sets up the optimization. This is where the input parameters are given as well as other run settings. Examples of scenario files can be found in tests/aux and the full template is found in baco/schema.json.

Running BaCO with a black-box function
To run it with a blackbox function, simply call the optimize() routine with a callable python-function and the name of the scenario file.

Running BaCO client-server
In the client-server mode, the compiler framework calls BaCO on demand asking for recommended settings.

The two parties communicate via a client (the third-party software) and server (BaCO) protocol defined by BaCO. The general idea of the protocol is that the client asks the server the configurations to run. So, there is the first design of experiments phase where samples are drawn and a second phase that is about the (Bayesian) optimization.

To enable the Client-Server mode add this line to the json file:

“baco_mode”: {
       “mode”: “client-server”
   }
The client and server communicate following a csv-like protocol via the standard output. The client calls BaCO to start the optimization process. When called by the client, BaCO will reply by requesting a number of function evaluations and wait for a reply. As an example, BaCO will reply with:

Request 3
x1,x2
-10,12
1,6
-8,20
Note that Hypermapper starts the protocol by stating how many evaluations it is requesting, followed by the input parameters (x1 and x2 int this case) and a series of parameter values.

Upon receiving this message, the client must compute the function values at the requested points and reply with the input parameters and the function values:

x1,x2,value,Valid
-10,12,267,False
1,6,28,True
-8,20,463,False
This protocol continues for the number of iterations specified by the client in the scenario file and after all iterations are done, Hypermapper will save all explored points to a csv file and end its execution.
