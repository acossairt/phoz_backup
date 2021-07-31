from deephyper.problem import HpProblem

Problem = HpProblem()

# Github version
#Problem.add_dim('units', (1, 100))
#Problem.add_dim('activation', ['identity', 'relu', 'sigmoid', 'tanh'])
#Problem.add_dim('lr', (0.0001, 1.))
# Website version
Problem.add_hyperparameter((1, 100), "units")
Problem.add_hyperparameter(["identity", "relu", "sigmoid", "tanh"], "activation")
Problem.add_hyperparameter((0.0001, 1.0), "lr")

Problem.add_starting_point(
    units=10,
    activation='identity',
    lr=0.01)

if __name__ == '__main__':
    print(Problem)
