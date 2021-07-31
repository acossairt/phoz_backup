from deephyper.problem import HpProblem

Problem = HpProblem()

Problem.add_hyperparameter((32, 1024), "units")
#Problem.add_hyperparameter(["identity", "relu", "sigmoid", "tanh"], "activation")
Problem.add_hyperparameter(["relu", "tanh"], "activation")
Problem.add_hyperparameter((1e-5, 1e-2), "lr")
# ntrain and ntest?
Problem.add_hyperparameter((50, 200), "nepochs")
Problem.add_hyperparameter((0.0, 1e-2), "dr")
Problem.add_hyperparameter((64, 1024), "batch_size") # different from units?

Problem.add_starting_point(units=64, activation="relu", lr=1e-4, nepochs=100, dr=1e-4, batch_size=256)

if __name__ == "__main__":
    print(Problem)
