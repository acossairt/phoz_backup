pip install 'deephyper[balsam,hvd]'
pip install 'deephyper[analytics]'

deephyper start-project hps_demo_phoz
cd hps_demo_phoz/hps_demo_phoz/
deephyper new-problem hps polynome2
cp ../../hps_demo_backup/hps_demo/polynome2/pho_z_* polynome2/
rm model_run.py problem.py load_data.py


###################### TO RUN ######################

deephyper hps ambs --problem hps_phoz.polynome2.pho_z_problem.Problem --run hps_phoz.polynome2.pho_z_model_run.run --max-evals 100

####################################################


To check baseline

Start python kernel

1. exec(open('pho_z_model_run.py').read())
2. point = {"units": 10, "activation": "relu", "lr": 0.0001, "nepochs": 20, "dr": 0.001, "batch_size": 256}
3. objective = run(point)
