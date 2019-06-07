import numpy as np
import ase2 as ase
import ase2.io as aio
from concurrent.futures import ProcessPoolExecutor
import time
import ase2.calculators.dftb as adftb
import qml as qml
import qml.representations as qmlrep
import scipy.spatial as sps


# Python library used for the simulation


class Trajectory:
    """docstring for trajectory"""

    def __init__(self, position_traj=[], energy_traj=[],
                 generation_details=None):
        self.position_traj = position_traj
        self.energy_traj = energy_traj
        self.generation_details = generation_details

    def extend(self, traj):
        if type(traj) is not type(self):
            raise ValueError('The input is not a trajectory')

        if traj.generation_details != self.generation_details:
            raise ValueError(
                'The trajectories to merge come from different simulations.')

        self.position_traj.extend(traj.position_traj)
        self.energy_traj.extend(traj.energy_traj)


class MCTrajectory:

    def __init__(self, position_traj=None, energy_traj=None, moves_used=None,
                 moves_accepted=None, generation_details=None,
                 flush_prefix=None):

        if position_traj is None:
            position_traj = []
        self.position_traj = position_traj

        if energy_traj is None:
            energy_traj = []
        self.energy_traj = energy_traj

        if generation_details is None:
            generation_details = {}
        self.generation_details = generation_details

        if moves_used is None:
            moves_used = []
        self.moves_used = moves_used

        if moves_accepted is None:
            moves_accepted = []
        self.moves_accepted = moves_accepted

    def extend(self, traj):
        if type(traj) is not type(self):
            raise ValueError('The input is not a trajectory')

        # if traj.generation_details != self.generation_details:
        #     raise ValueError(
        #         'The trajectories to merge come from different simulations.')

        self.position_traj.extend(traj.position_traj)
        self.energy_traj.extend(traj.energy_traj)
        self.moves_used.extend(traj.moves_used)
        self.moves_accepted.extend(traj.moves_accepted)

    def mc_probabilities(self):
        probabilities = []
        for i in range(len(self.generation_details['move_list'])):
            idxs = [t for t, x in enumerate(self.moves_used) if x == i]
            idxs_bool = [self.moves_accepted[t] for t in idxs]

            probabilities.append(sum(idxs_bool) / len(idxs_bool))
        return probabilities

    def flush(self, flush_prefix):
        if len(self.moves_used) > 0:
            f = open('{}_mc_moves.dat'.format(flush_prefix), 'ab')
            np.savetxt(f, np.array(
                list(zip(self.moves_used, self.moves_accepted))), fmt='%i')
            f.close()

        f = open('{}_energies.dat'.format(flush_prefix), 'ab')
        np.savetxt(f, np.array(self.energy_traj), fmt='%.6f')
        f.close()

        for struct in self.position_traj:
            aio.write('{}_structures.xyz'.format(flush_prefix),
                      ase.Atoms(self.generation_details['atoms'],
                                positions=struct), append=True)

        self.__init__(generation_details=self.generation_details,
                      flush_prefix=flush_prefix)


class DftbEnergy:
    """docstring for dftb"""

    def __init__(self, atoms, directory, **kwargs):
        self.dftb_kwargs = kwargs
        self.atoms = ase.Atoms(atoms)
        self.directory = directory
        self.calc = adftb.Dftb(**kwargs)
        self.calc.directory = directory

    def energy(self, structure):
        self.atoms.positions = structure
        self.calc.calculate(self.atoms)
        energy = self.calc.results['energy']
        ev_to_kcalmol = 23
        return energy * ev_to_kcalmol

    def force(self, structure):
        pass


class MixedPotential:
    """docstring for MixedPotential"""

    def __init__(self, energy_func1, energy_func2, alpha):
        self.energy_func1 = energy_func1
        self.energy_func2 = energy_func2

    def energy(self, structure):
        return self.energy_func1(
            structure) * (1 - self.alpha) + self.energy_func2(
            structure) * self.alpha


class KRR_potential:
    """docstring for ML_potential"""

    def __init__(self, representation_generator,
                 training_representations, alpha_values,
                 kernel, baseline=None, delta_scale=1):
        self.baseline = baseline
        self.representation_generator = representation_generator
        self.alpha_values = alpha_values
        self.kernel = kernel
        self.training_representations = training_representations
        self.delta_scale = delta_scale

    def energy(self, structure):
        delta_e = [0]
        if self.baseline is not None:
            ener = self.baseline(structure)
        else:
            ener = 0
        x = self.representation_generator.generate(structure)
        k_vec = self.kernel(np.expand_dims(x, axis=0),
                            self.training_representations)
        delta_e = self.delta_scale * np.dot(k_vec, self.alpha_values)
        return ener + delta_e[0]


class SLATMGenerator:
    """docstring for SLATMGenerator"""

    def __init__(self, atoms):
        self.atoms = atoms
        self.atomic_numbers = ase.Atoms(symbols=atoms).get_atomic_numbers()
        self.mbtypes = qml.representations.get_slatm_mbtypes(
            [self.atomic_numbers])

    def generate(self, structure):
        return qmlrep.generate_slatm(
            coordinates=structure, nuclear_charges=self.atomic_numbers,
            mbtypes=self.mbtypes)


class CMGenerator:
    """docstring for CMGenerator"""

    def __init__(self, atoms):
        self.atoms = atoms
        self.nuclear_charges = ase.Atoms(symbols=atoms).get_atomic_numbers()

    def generate(self, structure):

        return qmlrep.generate_coulomb_matrix(
            nuclear_charges=self.nuclear_charges,
            coordinates=structure,
            size=len(self.atoms))


class GaussianKernel:
    """docstring for GaussianKernel"""

    def __init__(self, sigma, norm=np.linalg.norm):

        self.norm = norm
        self.sigma

    def build(self, x, data):
        return np.exp(- (1 / self.sigma) * self.norm(data - x))


class GaussianVar:
    """docstring for GaussianVar"""

    def __init__(self, loc, var):

        self.loc = loc
        self.var = var

    def generate(self, size):
        return np.random.normal(self.loc, self.var, size)


class Reservoir:
    """docstring for Reservoir"""

    def __init__(self, structures, energies, temperature, energy_func,
                 kb=0.0019872041):
        self.structures = structures
        self.energies = energies
        self.size = len(energies)
        self.temperature = temperature
        self.beta = (kb * self.temperature) ** - 1
        self.energy_func = energy_func

    def simulation_type(self):

        return MCTrajectory(generation_details=self.simulation_details())

    def simulation_details(self):
        details = {'temperature': self.temperature,
                   'energy_func': self.energy_func}
        return details

    def flush(self):
        pass

    def run(self, *args):
        np.random.seed()
        empty = MCTrajectory(generation_details=self.simulation_details())
        idx = np.random.choice(np.arange(self.size))

        pos = self.structures[idx]
        ener = self.energies[idx]

        return [empty, pos, ener]


class MCSimulation:
    """docstring for MCSimulation"""

    def __init__(self, energy_func, temperature, atoms,
                 move_list, move_weight_list=None, kb=0.0019872041):

        self.temperature = temperature
        self.beta = (kb * self.temperature) ** - 1
        self.atoms = atoms
        self.energy_func = energy_func
        self.move_list = move_list
        self.move_weight_list = move_weight_list

    def simulation_details(self):
        return vars(self)

    def simulation_type(self):
        return MCTrajectory(generation_details=self.simulation_details())

    def _advance(self, old_pos, old_ener):

        move_idx = np.random.choice(
            list(range(len(self.move_list))), p=self.move_weight_list)
        move = self.move_list[move_idx]

        new_pos, new_ener, bias = move.move(
            old_position=old_pos, old_energy=old_ener, beta=self.beta)
        if new_ener is None:
            new_ener = self.energy_func(new_pos)

        new_weight = np.exp(- self.beta * new_ener)
        old_weight = np.exp(- self.beta * old_ener)

        prob = min([1, bias * new_weight / old_weight])

        accepted = np.random.rand() < prob
        # print((old_ener, new_ener))
        # print((prob, accepted))
        if accepted:
            return new_pos, new_ener, bias, move_idx, accepted
        else:
            return old_pos, old_ener, bias, move_idx, accepted

    def run(self, init_struct, steps, stride=10, init_ener=None,
            return_last=False):
        np.random.seed()
        pos = init_struct
        if init_ener is None:
            ener = self.energy_func(pos)
        else:
            ener = init_ener

        position_traj = []
        energy_traj = []
        moves_used = []
        moves_accepted = []
        bias_traj = []

        # append initial structure
        position_traj.append(pos)
        energy_traj.append(ener)

        for i in range(1, steps):
            pos, ener, bias, move_idx, accepted = self._advance(
                pos, ener)
            bias_traj.append(bias)
            moves_used.append(move_idx)
            moves_accepted.append(accepted)

            if i % stride == 0:
                position_traj.append(pos)
                energy_traj.append(ener)

        traj = MCTrajectory(position_traj, energy_traj, moves_used,
                            moves_accepted, self.simulation_details())
        if return_last is True:
            return [traj, pos, ener]

        else:
            return traj


class ReplicaExchangeSimulation:
    """docstring for ReplicaExchangeSimulation"""

    def __init__(self, num_reps, simulations, init_structs, stride, rep_steps,
                 reservoir=False, init_eners=None, directory='.'):
        # self.init_sumtrack = summary.summarize(muppy.get_objects())
        self.num_reps = num_reps

        if num_reps % 2 != 0:
            raise('Number of 00s must be pair')

        if len(simulations) != self.num_reps:
            raise('Wrong number of temperatures')

        self.temperatures = [sim.temperature for sim in simulations]

        self.energy_funcs = [sim.energy_func for sim in simulations]

        self.simulations = simulations

        self.init_rep_structs = init_structs

        self.par_exec = ProcessPoolExecutor(max_workers=num_reps)

        # print('e')
        if init_eners is None:
            pass
            self.init_rep_eners = list(self.par_exec.map(
                smap, self.energy_funcs, self.init_rep_structs))
        else:
            self.init_rep_eners = init_eners
        # print('e')
        self.rep_index = np.arange(self.num_reps)

        self.even_sims = self.rep_index[::2]

        self.odd_sims = self.rep_index[::2]

        self.accepted_exchanges = {(i, (i + 1) % self.num_reps):
                                   [] for i in range(self.num_reps)}

        self.strides = [stride for i in range(num_reps)]

        self.rep_steps = rep_steps

        for stride in self.strides:
            if self.rep_steps % stride != 0:
                raise ValueError('Rep_steps must be multiple of stride')

        self.rep_stepss = [rep_steps for i in range(self.num_reps)]

        self.directory = directory

    def run(self, num_exchanges):

        trajectories = [sim.simulation_type() for sim in self.simulations]
        for i in range(num_exchanges):

            t0 = time.time()

            # generate dynamics
            # run individual simulation in parallel
            return_last = [True for l in range(self.num_reps)]

            simulation_results = list(
                self.par_exec.map(run_simulation, self.simulations,
                                  self.init_rep_structs, self.rep_stepss,
                                  self.strides, self.init_rep_eners,
                                  return_last))

            rep_trajs = [res[0] for res in simulation_results]
            exchange_structs = [res[1] for res in simulation_results]
            exchange_eners = [res[2] for res in simulation_results]

            for k in range(self.num_reps):
                trajectories[k].extend(rep_trajs[k])

            aaa, bbb = self._replica_exchange(exchange_structs, exchange_eners)

            self.init_rep_structs = aaa
            self.init_rep_eners = bbb

            self.exchange_probabilities = {key: (0.001 + sum(val)) / (len(
                val) + 0.001) for key, val in self.accepted_exchanges.items()}

            if i % 2 == 1:
                for rep, traj in enumerate(trajectories):
                    traj.flush(flush_prefix=(
                        self.directory + '/hrem.rep{}_'.format(rep)))

            t1 = time.time()

            with open("exchange.txt", "a") as myfile:
                myfile.write(
                    'Exchange {0}, step {1}, time interval {2:.3} \n'.format(
                        i + 1, (i + 1) * self.rep_steps, t1 - t0))
                [myfile.write('{0}: {1:.3}\n'.format(
                    x, y)) for x, y in self.exchange_probabilities.items()]

    def _replica_exchange(self, exchange_structs, exchange_eners):
        shift = np.random.choice([1, -1])

        rep_index = np.arange(self.num_reps)

        group1 = rep_index[::2]
        group2 = rep_index[1::2]

        if shift == 1:
            ex_index = np.vstack((group2, group1)).flatten(order='F')

        else:

            ex_index = np.roll(
                np.vstack((group1, np.roll(group2, 1))).flatten(
                    order='F'), -1)

        pairs = list(zip(group1, ex_index[::2]))

        old_structs = exchange_structs
        old_energies = exchange_eners

        new_structs = [old_structs[i] for i in ex_index]

        new_energies = list(self.par_exec.map(
            smap, self.energy_funcs, new_structs))

        with open("log.txt", "a") as myfile:
            myfile.write('================================')
            myfile.write('Exchange')
            myfile.write('================================')

        for pair in pairs:

            rep0 = self.simulations[pair[0]]
            rep1 = self.simulations[pair[1]]

            old_e0 = old_energies[pair[0]]
            old_e1 = old_energies[pair[1]]

            new_e0 = new_energies[pair[0]]
            new_e1 = new_energies[pair[1]]


            old_weight = rep0.beta * old_e0 + rep1.beta * old_e1
            new_weight = rep0.beta * new_e0 + rep1.beta * new_e1

            prob = mc_prob(weight_new=new_weight, weight_old=old_weight)
            accepted = np.random.rand() < prob

            with open("log.txt", "a") as myfile:
                myfile.write('\n')
                myfile.write('Rep A: ')
                myfile.write('{}'.format(pair[0]))
                myfile.write('\n')
                myfile.write('Old Energy: ')
                myfile.write('{0:.5f} '.format(old_e0))
                myfile.write('\n')
                myfile.write('New Energy: ')
                myfile.write('{0:.5f} '.format(new_e0))
                myfile.write('\n')
                myfile.write('beta rep A: ')
                myfile.write('{0:.5f} '.format(rep0.beta))
                myfile.write('\n')

                myfile.write('Rep B: ')
                myfile.write('{}'.format(pair[1]))
                myfile.write('\n')
                myfile.write('Old Energy: ')
                myfile.write('{0:.5f} '.format(old_e1))
                myfile.write('\n')
                myfile.write('New Energy: ')
                myfile.write('{0:.5f} '.format(new_e1))
                myfile.write('\n')
                myfile.write('beta rep B: ')
                myfile.write('{0:.5f} '.format(rep1.beta))
                myfile.write('\n')
                myfile.write('Old weight: ')
                myfile.write('{0:.5f} '.format(old_weight))
                myfile.write('\n')
                myfile.write('New weight: ')
                myfile.write('{0:.5f} '.format(new_weight))
                myfile.write('\n')
                myfile.write('Exchange Prob: ')
                myfile.write('{0:.5f} '.format(prob))
                myfile.write('Accepted: ')
                myfile.write('{} '.format(bool(accepted)))
                myfile.write('\n')
                myfile.write('---------------------------------------------')
                myfile.write('\n')


            if shift == 1:
                self.accepted_exchanges[(pair[0], pair[1])].append(accepted)
            else:
                self.accepted_exchanges[(pair[1], pair[0])].append(accepted)

            if accepted:
                pass
            else:
                new_structs[pair[0]] = old_structs[pair[0]]
                new_structs[pair[1]] = old_structs[pair[1]]

                new_energies[pair[0]] = old_energies[pair[0]]
                new_energies[pair[1]] = old_energies[pair[1]]
        return new_structs, new_energies


def mc_accept(weight_new, weight_old):

    exp = np.exp(- weight_new + weight_old)

    if exp > np.random.rand():
        return True

    else:
        return False


def mc_prob(weight_new, weight_old):

    prob = min([1, np.exp(- weight_new + weight_old)])

    return prob


def run_simulation(simulation, *args):
    return simulation.run(*args)


def smap(f, *args):
    return f(*args)


def _advance_mc(old_pos, old_ener, energy_func, beta, move_list,
                move_weights=None):

    idx_move = np.random.choice(
        list(range(len(move_list))), p=move_weights)
    move = move_list[idx_move]

    new_pos = move.move(old_pos)
    new_ener = energy_func(new_pos)

    new_weight = beta * new_ener
    old_weight = beta * old_ener

    prob = mc_prob(weight_new=new_weight, weight_old=old_weight)
    accepted = np.random.rand() < prob

    if accepted:
        return new_pos, new_ener, idx_move, accepted
    else:
        return old_pos, old_ener, idx_move, accepted


def run_mc(init_struct, init_ener, temperature, energy_func, steps,
           move_list, move_weights=None, stride=10,
           kb=0.0019872041, rex=True):

    np.random.seed()

    struct_traj = []
    ener_traj = []
    idx_moves = []
    moves_acc = []

    beta = (kb * temperature) ** -1
    pos = init_struct
    ener = init_ener
    for i in range(1, steps):
        if i % stride == 0:
            struct_traj.append(pos)
            ener_traj.append(ener)

        pos, ener, idx_move, accepted = _advance_mc(
            pos, ener, energy_func, beta, move_list, move_weights)

        idx_moves.append(idx_move)
        moves_acc.append(accepted)

    last_struc = pos
    last_ener = ener
    if rex is True:
        return struct_traj, ener_traj, idx_moves, moves_acc, \
            last_struc, last_ener
    else:
        return struct_traj, ener_traj, idx_moves, moves_acc


class HamiltonianMCMove:
    """docstring for HybridMCMove"""

    def __init__(self, propagator, md_steps, temperature):

        self.propagator = propagator
        self.molecule = propagator.molecule
        self.calculator = propagator.molecule.get_calculator()
        self.masses = propagator.molecule.get_masses()
        self.temperature = temperature

    def move(self, old_position, beta, old_ener=None, ** kwargs):

        self.molecule.positions = old_position
        ase.md.velocitydistribution.MaxwellBoltzmannDistribution(
            self.molecule, temp=300. * ase.units.kB)

        # if old_ener is None:
        #     self.calculator.calculate(self.molecule)
        #     old_pot = self.molecule.get_potential_energy()
        # else:
        #     old_pot = old_ener

        old_kin = self.molecule.get_kinetic_energy()

        init_velocities = self.molecule.get_velocities()

        new_pos, new_pot, final_velocities = self.propagator.propagate(
            old_position, init_velocities)

        self.molecule.set_velocities(final_velocities)

        new_kin = self.molecule.get_kinetic_energy()

        # old_H = old_pot + old_kin
        # new_H = new_pot + new_kin

        bias = np.exp(- self.beta * (new_kin - old_kin))

        return new_pos, new_pot, bias


class MTSMCMove:
    """docstring for MTSMC"""

    def __init__(self, cheap_MC_simulation, chain_length):

        self.temperature = cheap_MC_simulation.temperature
        self.beta = cheap_MC_simulation.beta
        self.atoms = cheap_MC_simulation.atoms

        self.cheap_mc_sim = cheap_MC_simulation

        self.cheap_potential = cheap_MC_simulation.energy_func
        self.chain_length = chain_length

    def move(self, old_position, **kwargs):
        old_cheap_energy = self.cheap_potential(old_position)

        traj, new_position, new_cheap_energy = self.cheap_mc_sim.run(
            init_struct=old_position,
            steps=self.chain_length,
            init_ener=old_cheap_energy,
            stride=9999, return_last=True)

        bias = np.exp(self.beta * (new_cheap_energy - old_cheap_energy))

        new_expensive_ener = None

        return new_position, new_expensive_ener, bias


class MDVerletPropagator:
    """docstring for DFTBMDpropagator"""

    def __init__(self, atoms, calculator, time_step=1):
        self.molecule = ase.Atoms(atoms)
        self.molecule.set_calculator(calculator)

    def propagate(self, structure, init_velocities, md_steps,
                  return_velocities=False):

        self.molecule.set_positions(structure)
        self.molecule.set_velocities(init_velocities)

        dyn = ase.md.VelocityVerlet(
            self.molecule, self.time_step * ase.units.fs)

        dyn.run(md_steps)

        if return_velocities is False:
            return self.molecule.positions, \
                self.molecule.get_potential_energy()
        else:
            return self.molecule.positions, \
                self.molecule.get_potential_energy(), \
                self.molecule.get_velocities()


class KRRGradient:
    '''Class that compute the force and the potential for any representation
        with the Gaussian Kernel.
    '''

    def __init__(self, training_set, gamma, alphas, num_atoms, delta_scale=1, baseline=None):
        self.training_set = training_set
        # training set is the D matrix
        self.gamma = gamma
        self.alphas = alphas
        self.num_atoms = num_atoms
        self.num_coordinates = 3
        self.baseline = baseline
        self.delta_scale = delta_scale

    def compute(self, input_representation):
        '''Compute the predicted force for an input representation knowing the
        training_set and the gamma value
         '''
        # input_representation is the M matrix representation
        input_representation.generate_gradient()
        rep_vector = input_representation.rep_vector

        diffs = rep_vector - self.training_set
        # print('diffs: ', diffs[0])
        # print(np.array([np.linalg.norm(
        #     rep_vector - x) for x in self.training_set]))

        norms = np.linalg.norm(diffs, axis=1)
        # print(norms)

        exponential_vector = np.exp(- self.gamma * norms ** 2)
        # print(exponential_vector)

        potential = np.sum(exponential_vector * self.alphas)
        # print(potential)
        # exponential vector that come from the Kernel

        force = np.zeros([self.num_atoms, 3])
        for atomes in range(self.num_atoms):
            for coordinates in range(self.num_coordinates):

                grad_vector = input_representation.grad_vector(
                    atomes, coordinates)

                vector_sum = np.sum(diffs * grad_vector, axis=1)
                force[atomes][coordinates] = np.sum(
                    exponential_vector * 2 * self.alphas * self.gamma *
                    vector_sum)

        # potential += baseline.potential(
        #                   input_representation.coordinates())
        # force += baseline.force(input_representation.coordinates())

        if self.baseline is not None:

            mol = ase.Atoms(input_representation.input_charge)
            mol.set_positions(input_representation.input_structure)
            mol.set_calculator(self.baseline)

            baseline_energy = mol.get_potential_energy()
            baseline_force = mol.get_forces()

        else:

            baseline_energy = 0
            baseline_force = 0

        return self.delta_scale*potential + baseline_energy, self.delta_scale*force + baseline_force

    # the function energy(), compute the potential for an input representation
    # knowing the training_set and the gamma value

    def energy(self, input_representation):
        '''Input_representation is the M matrix representation'''

        rep_vector = input_representation.rep_vector

        diffs = - self.training_set + rep_vector

        norms = np.linalg.norm(diffs, axis=1)

        exponential_vector = np.exp(- self.gamma * norms**2)

        potential = np.sum(exponential_vector * self.alphas)

        if self.baseline is not None:

            mol = ase.Atoms(input_representation.input_charge)
            mol.set_positions(input_representation.input_structure)
            mol.set_calculator(self.baseline)
            
            self.baseline.calculate(mol)

            baseline_energy = mol.get_potential_energy()

        else:

            baseline_energy = 0

        return self.delta_scale*potential + baseline_energy


class CoulombMatrix:

    '''Class that generates the Coulomb Matrix (CM)representation and its
    derivative with respect to atomic coordinate'''

    def __init__(self, input_structure, input_charge):

        self.num_atoms = input_structure.shape[0]
        self.input_charge = input_charge
        self.input_structure = input_structure
        self.num_coordinates = 3
        self.rep_vector = self.generate_representation()

    '''the generate_representation(), generate the CM vector representation'''

    def generate_representation(self):
        Z_outer_matrix = np.outer(
            self.input_charge, self.input_charge).astype(float)

        np.fill_diagonal(Z_outer_matrix, 0.5 *
                         np.power(self.input_charge, 2.4))

        Z_final_matrix = Z_outer_matrix

        atomic_distances = sps.distance_matrix(
            self.input_structure, self.input_structure) + np.identity(
            self.num_atoms)

        inv_atomic_distances = 1 / atomic_distances
        representation = Z_final_matrix * inv_atomic_distances
        indexlisted = np.argsort(np.linalg.norm(representation, axis=1))

        self.rep_matrix = representation
        flat_rep = representation[np.tril_indices(representation.shape[0])]

        return flat_rep

    def generate_gradient(self):
        atomic_distances = sps.distance_matrix(
            self.input_structure, self.input_structure)
        grad_M = np.zeros(
            shape=[self.num_atoms, 3, self.num_atoms, self.num_atoms])
        for atom in range(self.num_atoms):
            for coordinates in range(self.num_coordinates):
                for i in range(atom + 1, self.num_atoms):

                    val = ((
                        self.input_structure[i][coordinates] -
                        self.input_structure[atom][coordinates]) *
                        (self.input_charge[i] * self.input_charge[atom])) / \
                        (atomic_distances[i][atom]**3)

                    grad_M[atom][coordinates][atom][i] = val
                    grad_M[atom][coordinates][i][atom] = val
                    grad_M[i][coordinates][i][atom] = -val
                    grad_M[i][coordinates][atom][i] = -val

        self.grad_M = grad_M
        return grad_M

    # the grad_vector(), generate the CM vector representation
    # according to each atoms and atomic coordinates

    def grad_vector(self, atom, coordinate):

        dm_dx = self.grad_M[atom][coordinate]

        dm_dx = dm_dx[np.tril_indices(dm_dx.shape[0])]
        return dm_dx


class VelocityVerletKRRPotentialSimulation:
    # Class that propagate the atomics positions by a Velocity Verlet
    # algorithme for any respresentation class

    def __init__(self, time_step, atoms, KRR_force_model, representation_class,
                 langevin_thermostat=False, langevin_friction_coeff=10,
                 temperature=300, verbose=False, kb=0.0019872041):

        self.atoms = atoms
        self.time_step = time_step
        self.ase_molecule = ase.Atoms(atoms)
        self.masses = self.ase_molecule.get_masses()
        self.charges = self.ase_molecule.get_atomic_numbers()
        self.krr_force = KRR_force_model
        self.representation_class = representation_class
        self.langevin_thermostat = langevin_thermostat
        self.langevin_friction_coeff = langevin_friction_coeff
        self.temperature = temperature
        self.beta = (kb * self.temperature) ** - 1
        self.verbose = verbose

    def energy_func(self, struct):
        ev_to_kcalmol = 23
        return self.krr_force.energy(
            self.representation_class(struct, self.charges)) * ev_to_kcalmol

    def simulation_details(self):
        return vars(self)

    def simulation_type(self):
        return MCTrajectory(generation_details=self.simulation_details())

    def run(self, init_struct, steps, stride=10, init_ener=None,
            return_last=False):
        np.random.seed()
        input_velocity = maxwell_boltzmann_distribution(
            self.atoms, self.temperature)

        langevin_thermostat = self.langevin_thermostat
        langevin_friction_coeff = self.langevin_friction_coeff
        temperature = self.temperature
        verbose = self.verbose

        numb_iterations = steps
        positions = []
        representations = []
        velocity = []
        T = []
        times = []
        accelerations = []
        potential_energies = []
        kinetic_energies = []
        total_energies = []
        forces = []
        boltzmann_constant = 1.38064852 * 1e-23
        amu_to_kg = 1.660540199 * 1e-27
        # avogadro_constant = 6.02214086 * 1e-23
        ev_to_joule = 1.602176565 * 1e-19
        # angstfs_to_ms = 1e5
        ev_to_kcalmol = 23
        ms_to_angstfs = 1e-5
        # joule_to_ev = 1 / ev_to_joule

        positions.append(init_struct)

        velocity.append(input_velocity)

        representation = self.representation_class(
            init_struct, self.charges)

        potential, force = self.krr_force.compute(representation)

        representations.append(representation)

        forces.append(force)

        masses_kg = self.masses * amu_to_kg

        inverse_masses = np.array(1 / masses_kg)

        force = force * ev_to_joule * (ms_to_angstfs**2)

        acceleration = force * inverse_masses[:, np.newaxis]

        accelerations.append(acceleration)

        velocity.append(velocity[0] + accelerations[0] * self.time_step)

        numb_iterations = int(numb_iterations)

        for i in range(0, numb_iterations-1):
            if langevin_thermostat:

                t1 = time.time()

                coeff1 = (2 - langevin_friction_coeff * self.time_step) / \
                    (2 + langevin_friction_coeff * self.time_step)

                coeff2 = 1e-5 * np.sqrt(boltzmann_constant * temperature *
                                        self.time_step * 0.5 *
                                        langevin_friction_coeff / masses_kg)

                coeff3 = 2 * self.time_step / \
                    (2 + langevin_friction_coeff * self.time_step)

                eta = np.random.normal(0, 1, (len(self.atoms), 3))

                vel_half_step = velocity[i] + accelerations[i] * \
                    0.5 * self.time_step + coeff2[:, np.newaxis] * eta

                new_position = positions[i] + coeff3 * vel_half_step

                positions.append(new_position)

                generation_representation = self.representation_class(
                    new_position, self.charges)
                potential, generation_force = self.krr_force.compute(
                    generation_representation)
                forces.append(generation_force)
                potential_energies.append(potential)

                scaled_force = generation_force * \
                    ev_to_joule * (ms_to_angstfs**2)
                generation_acceleration = scaled_force * \
                    inverse_masses[:, np.newaxis]

                accelerations.append(generation_acceleration)

                velocity.append(coeff1 * vel_half_step +
                                coeff2[:, np.newaxis] * eta + 0.5 * (
                                    accelerations[i + 1]) * self.time_step)

                times.append(self.time_step + self.time_step * i)

            else:

                t1 = time.time()
                positions.append(
                    positions[i] + velocity[i] * self.time_step + (
                        accelerations[i] * (self.time_step**2) * 0.5))

                new_position = positions[i + 1]

                generation_representation = self.representation_class(
                    new_position, self.charges)
                potential, generation_force = self.krr_force.compute(
                    generation_representation)
                forces.append(generation_force)
                potential_energies.append(potential)

                generation_force = generation_force * \
                    ev_to_joule * (ms_to_angstfs**2)

                generation_acceleration = generation_force * \
                    inverse_masses[:, np.newaxis]

                accelerations.append(generation_acceleration)

                velocity.append(velocity[i] + 0.5 * (
                    accelerations[i] + accelerations[i + 1]) * self.time_step)

                times.append(self.time_step + self.time_step * i)

            kinetic_energy_ev = 0.5 * (1 / (0.098227023)**2) * np.vdot(
                velocity[i + 1] * np.array(
                    self.masses)[:, np.newaxis], velocity[i + 1])

            kinetic_energies.append(kinetic_energy_ev)

            total_energy = potential + kinetic_energy_ev
            total_energies.append(total_energy)

            number_degrees_freedom = 3 * len(self.masses)
            T_inst = (2 * kinetic_energy_ev) / \
                (number_degrees_freedom * 8.6173303 * 1e-5)
            T.append(T_inst)

            if verbose is True:
                print('Time of simulation:', '    ', times[i], 'fs')
                print('')
                print('')
                print('Potential energy:', '    ', potential_energies[i], 'eV')
                print('Kinetic energy:', '      ', kinetic_energies[i], 'eV')
                print('Total energy:', '        ', total_energies[i], 'eV')
                print('')
                print('Instantneous temperature:', T_inst, 'K')
                print('')
                print('')

                print('END OF THE', i, '-th ITERATIONS')
                print('Time cost', time.time() - t1)

                print('')
                print('')
            with open("inst_temperature.txt", "a") as myfile:
                myfile.write('{0:.5f} '.format(T_inst))


        if verbose is True:
            print('Average temperature', '            ', sum(T) / len(T), 'K')
            print('Final Total potential energy:',
                  '    ', potential_energies[-1], 'eV')
            print('Final Total kinetic energy:',
                  '      ', kinetic_energies[-1], 'eV')
            print('Final Total energy:', '              ',
                  total_energies[-1], 'eV')

        potential_energies = np.array(potential_energies) * ev_to_kcalmol

        traj = MCTrajectory(position_traj=positions[::stride],
                            energy_traj=potential_energies[::stride],
                            generation_details=self.simulation_details(),
                            moves_used=None, moves_accepted=None)
        if return_last is True:
            return [traj, positions[-1], potential_energies[-1]]

        else:
            return positions


def maxwell_boltzmann_distribution(atoms, temperature):
    input_velocities = np.zeros(shape=[len(atoms), 3])
    T = temperature
    kb = 1.38064852 * 1e-23
    Na = 6.02214086 * 1e23

    masse = [1e-3 * M / Na for M in ase.Atoms(atoms).get_masses()]
    standard_deviation = [np.sqrt((kb * T) / m) for m in masse]

    for i in range(len(standard_deviation)):
        for j in range(3):
            input_velocities[i][j] = 1e-5 * np.random.normal(
                loc=0, scale=standard_deviation[i], size=[1])
    return input_velocities
