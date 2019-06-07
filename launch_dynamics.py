import numpy as np
import ase2 as ase
import ase2.io as aio

import mc_lib as mcl
from ase2.calculators.dftb import Dftb


# Script to launch hamiltonian replica exchange machine learning potential


# Machine learning model data
charges, X_training, alphas, sigma = np.load('./ML_model_data.npy')
# Structures to start the dynamic
structures = aio.read('reservoir_structures_baseline.xyz', index=':')
# Number of replicas
num_reps = 4
# structures = []
# for i in range(num_reps-1):
#      structures.append(aio.read('./run1/hrem.rep{}__structures.xyz'.format(i), index='-1'))

# build KRR potential
atoms = structures[0].get_chemical_symbols()
num_atoms = len(atoms)

dftb_baselines = [Dftb(
     atoms=ase.Atoms(atoms),
     run_manyDftb_steps=False,
     Hamiltonian_ReadInitialCharges='No',
     Hamiltonian_MaxSCCIterations=100,
     Hamiltonian_DampXH='Yes',
     Hamiltonian_DampXHExponent=4.0,
     Hamiltonian_Eigensolver='RelativelyRobust{}',
     Hamiltonian_SCC='Yes',
     Hamiltonian_ThirdOrderFull='Yes',
     Hamiltonian_SCCTolerance=1.0E-008,
     Hamiltonian_Dispersion_='SlaterKirkwood',
     Hamiltonian_Dispersion_PolarRadiusCharge_='HybridDependentPol',
     Hamiltonian_Dispersion_PolarRadiusCharge_H='{\n CovalentRadius[Angstrom] = 0.4 \n HybridPolarisations [Angstrom^3,Angstrom,] = {\n0.386 0.396 0.400 0.410 0.410 0.410 3.5 3.5 3.5 3.5 3.5 3.5 0.8 \n }\n }',
     Hamiltonian_Dispersion_PolarRadiusCharge_C='{\n CovalentRadius[Angstrom] = 0.76 \n HybridPolarisations [Angstrom^3,Angstrom,] = {\n1.382 1.382 1.382 1.064 1.064 1.064 3.8 3.8 3.8 3.8 3.8 3.8 2.50\n }  \n}',
     Hamiltonian_Dispersion_PolarRadiusCharge_S='{\n CovalentRadius[Angstrom] = 1.02 \n HybridPolarisations [Angstrom^3,Angstrom,] = {\n3.000 3.000 3.000 3.000 3.000 3.000 4.7 4.7 4.7 4.7 4.7 4.7 4.80\n }  \n}',

#    Modification if one want to use DFTB dDMC as baseline
     # Hamiltonian_Dispersion_='dDMC{}',
     # Hamiltonian_Filling_ = 'Fermi',
     # Hamiltonian_Filling_Fermi_Temperature='0.0009500372557109826',
     Hamiltonian_MaxAngularMomentum_='',
     Hamiltonian_MaxAngularMomentum_C='"p"',
     Hamiltonian_MaxAngularMomentum_H='"s"',
     Hamiltonian_MaxAngularMomentum_S='"d"',
     Hamiltonian_HubbardDerivs_='',
     Hamiltonian_HubbardDerivs_C='-0.1492',
     Hamiltonian_HubbardDerivs_H='-0.1857',
     Hamiltonian_HubbardDerivs_S='-0.11',
) for i in range(num_reps-1)]

for i, calc in enumerate(dftb_baselines):
     calc.directory = './dftb_rep_{}'.format(i)

dftb_reservoir = mcl.DftbEnergy(
    atoms=ase.Atoms(atoms), directory='./dftb_rep_{}'.format(num_reps),
    run_manyDftb_steps=False,
    Hamiltonian_ReadInitialCharges='No',
    Hamiltonian_MaxSCCIterations=100,
    Hamiltonian_DampXH='Yes',
    Hamiltonian_DampXHExponent=4.0,
    Hamiltonian_Eigensolver='RelativelyRobust{}',
    Hamiltonian_SCC='Yes',
    Hamiltonian_ThirdOrderFull='Yes',
    Hamiltonian_SCCTolerance=1.0E-008,
    Hamiltonian_Dispersion_='SlaterKirkwood',
    Hamiltonian_Dispersion_PolarRadiusCharge_='HybridDependentPol',
    Hamiltonian_Dispersion_PolarRadiusCharge_H='{\n CovalentRadius[Angstrom] = 0.4 \n HybridPolarisations [Angstrom^3,Angstrom,] = {\n0.386 0.396 0.400 0.410 0.410 0.410 3.5 3.5 3.5 3.5 3.5 3.5 0.8 \n }\n }',
    Hamiltonian_Dispersion_PolarRadiusCharge_C='{\n CovalentRadius[Angstrom] = 0.76 \n HybridPolarisations [Angstrom^3,Angstrom,] = {\n1.382 1.382 1.382 1.064 1.064 1.064 3.8 3.8 3.8 3.8 3.8 3.8 2.50\n }  \n}',
    Hamiltonian_Dispersion_PolarRadiusCharge_S='{\n CovalentRadius[Angstrom] = 1.02 \n HybridPolarisations [Angstrom^3,Angstrom,] = {\n3.000 3.000 3.000 3.000 3.000 3.000 4.7 4.7 4.7 4.7 4.7 4.7 4.80\n }  \n}',

#    Modification if one want to use DFTB dDMC as baseline
    # Hamiltonian_Dispersion_='dDMC{}',
    # Hamiltonian_Filling_ = 'Fermi',
    # Hamiltonian_Filling_Fermi_Temperature='0.0009500372557109826',
    Hamiltonian_MaxAngularMomentum_='',
    Hamiltonian_MaxAngularMomentum_C='"p"',
    Hamiltonian_MaxAngularMomentum_H='"s"',
    Hamiltonian_MaxAngularMomentum_S='"d"',
    Hamiltonian_HubbardDerivs_='',
    Hamiltonian_HubbardDerivs_C='-0.1492',
    Hamiltonian_HubbardDerivs_H='-0.1857',
    Hamiltonian_HubbardDerivs_S='-0.11',
)

# Simulation properties
gamma = 1 / (2 * sigma ** 2)
krrs = [mcl.KRRGradient(X_training, gamma=gamma,
                         alphas=alphas, num_atoms=num_atoms,
                         delta_scale=(1 - i * 1 / (num_reps - 1)),baseline=dftb_baselines[i]) for i in range(num_reps-1)]
repr_class = mcl.CoulombMatrix

simulations = [mcl.VelocityVerletKRRPotentialSimulation(
     time_step=0.4, atoms=atoms, KRR_force_model=krrs[i],
     representation_class=repr_class,
     langevin_thermostat=True, langevin_friction_coeff=1,
     verbose=False,
     temperature=300) for i in range(num_reps-1)]

# Reservoir datas
reserv_ase = aio.read('reservoir_structures_baseline.xyz', index=':')
reserv_energies = np.loadtxt('reservoir_energy_baseline_eV.dat')
reserv_structs = [x.positions for x in reserv_ase]

reservoir = mcl.Reservoir(structures=reserv_structs, energies=reserv_energies,
                          temperature=300, energy_func=dftb_reservoir.energy)

simulations.append(reservoir)

# Initial structures of each replicas
init_idx = np.random.choice(np.arange(len(reserv_structs)), size=num_reps)
init_structures = [reserv_structs[i] for i in init_idx]


# Create replica exchange simulation
rep_rex = mcl.ReplicaExchangeSimulation(
     num_reps=num_reps, simulations=simulations,
     init_structs=init_structures,
     stride=25, rep_steps=50)

# Launch dynamics
rep_rex.run(20000)
