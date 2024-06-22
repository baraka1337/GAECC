# Project README

Welcome to the GAECC project!

This project, GAECC (Genetic Algorithm for Error Correction Codes), is designed to find the optimal linear code block systematic matrix by leveraging the power of Genetic Algorithm. The goal is to minimize the bit error rate in a given channel.

## How it works

The Genetic Algorithm implemented in this project follows a population-based approach to evolve and optimize the code block systematic matrix. It starts with an initial population of candidate matrices and iteratively applies genetic operators such as selection, crossover, and mutation to generate new generations. The fitness of each candidate matrix is evaluated based on its performance in terms of the bit error rate.

## Getting started

To get started with the GAECC project, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the GAECC folder.
3. Install the required dependencies by running the following command:
```
pip install -r requirements.txt
```
## Running tests

To run the tests for the GAECC project, you can use the `rerun_test.sh` script. This script automates the process of running the test cases and provides a convenient way to validate the functionality of the code.

The test cases are defined in the `test_template.py` file. This file contains various test scenarios that cover different aspects of the code's behavior. By running the tests, you can ensure that the code is functioning correctly and producing the expected results.

In addition to validating the functionality of the code, the `test_template.py` file also includes useful functions to analyze the algorithm's performance. You can leverage these functions to gain insights into the performance of the genetic algorithm and make any necessary improvements.

To execute the tests, simply run the following command in your terminal:

```
./rerun_test.sh
```

This will trigger the execution of the test cases and display the results in the terminal. Make sure to review the output to ensure that all tests pass successfully.

Feel free to modify the `test_template.py` file to add additional test cases or customize the existing ones according to your requirements.

Happy coding!