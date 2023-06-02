import numpy as np
import scipy
import qib
import h5py
import rqcopt as oc
import matplotlib.pyplot as plt

import numpy as np
from qib.field import ParticleType, Field
from qib.lattice import LayeredLattice
from qib.operator import AbstractOperator, FieldOperator, FieldOperatorTerm, IFOType, IFODesc


# TODO: possibility of non-uniform parameters?

def trotterized_time_evolution(L: int, hloc, perm_set, method: oc.SplittingMethod, dt: float, nsteps: int):
    """
    Compute the numeric ODE flow operator of the quantum time evolution
    based on the provided splitting method.
    """
    Vlist = []
    perms = []
    for i, c in zip(method.indices, method.coeffs):
        Vlist.append(scipy.linalg.expm(-1j*c*dt*hloc[i]))
        perms.append(perm_set[i])
    V = oc.brickwall_unitary(Vlist, L, perms)
    return np.linalg.matrix_power(V, nsteps)

def construct_fermi_hubbard_local_kinetic_term(J):
    return -J * np.array([[0., 0., 0., 0.], [0., 0., 1., 0.], [0., 1., 0., 0.], [0., 0., 0., 0.]]);

def construct_fermi_hubbard_local_interaction_term(U):
    n_spin = np.array([[0., 0.], [0., 1.]])
    return U * np.kron(n_spin, n_spin);

def construct_fermi_hubbard_dynamics_opt(nlayers: int, method_start: oc.SplittingMethod=None , **kwargs):
    # side length of lattice
    # leave even
    L = 4

    # Hamiltonian parameters
    J = 1.
    # TODO: obeserve effect of bigger U (e.g. >= 4)
    U = 4

    # time 
    # TODO: obeserve effect of changing time step (e.g. 1/8)
    t = 1/8
    print("t:", t)

    # construct Hamiltonian
    # latt = qib.lattice.LayeredLattice(qib.lattice.IntegerLattice((L,), pbc=False), 2)
    # field = qib.field.Field(qib.field.ParticleType.FERMION, latt)
    # H = FermiHubbardHamiltonianMod(field, J, U, True).as_matrix()
    
    H = np.zeros((2**(2*L), 2**(2*L)))

    n_spin = np.eye(2)
    n_spin[0,0] = 0
    n_spin_big = np.kron(n_spin, np.kron(np.eye(2 ** (L - 1)), n_spin))
    for i in range (L):
        H += U * np.kron(np.eye(2 ** i), np.kron(n_spin_big, np.eye(2 ** (L - i - 1))))

    ### LOCAL KINETIC TERM
    hloc_horz = construct_fermi_hubbard_local_kinetic_term(J)
    a_i = np.zeros((2,2))
    a_i1 = np.zeros((2,2))
    a_i[1,0] = 1
    a_i1[0,1] = 1
    for i in range (L - 1):
        H += np.kron(np.eye(2 ** i), np.kron(hloc_horz, np.eye(2**(2 * L - i - 2))))
    for i in range (L, 2*L - 1):
        H += np.kron(np.eye(2 ** i), np.kron(hloc_horz, np.eye(2**(2 * L - i - 2))))
    H += -J * np.kron(a_i1, np.kron(np.eye(2**(L - 2)), np.kron(a_i, np.eye(2**(L)))))
    H += -J * np.kron(a_i, np.kron(np.eye(2**(L - 2)), np.kron(a_i1, np.eye(2**(L)))))
    H += -J * np.kron(np.kron(np.eye(2**(L)), a_i1), np.kron(np.eye(2**(L - 2)), a_i))
    H += -J * np.kron(np.kron(np.eye(2**(L)), a_i), np.kron(np.eye(2**(L - 2)), a_i1))

    print("H.shape:", H.shape)

    # reference global unitary
    expiH = scipy.linalg.expm(-1j*H*t)

    hloc_vert = construct_fermi_hubbard_local_interaction_term(U)
    hloc = [hloc_horz, hloc_horz, hloc_vert]
    assert len(method_start.coeffs) == nlayers

    horz_even_sites = None

    # TODO: reconsider different notation (2x np.roll)
    horz_odd_sites = np.roll(range(L*2), -1)
    horz_odd_sites[[L - 1, L*2 -1]] = horz_odd_sites[[L*2 - 1, L -1]]
    print(horz_odd_sites)
    vert_sites = np.array([[i, i + L] for i in range(L)]).flatten()

    # Vlist_start = [scipy.linalg.expm(-1j*c*t*hloc) for c in coeffs_start]
    # perms = [None if i % 2 == 0 else np.roll(range(2*L), -1) for i in range(len(Vlist_start))]
    perm_set = [
        horz_even_sites, 
        list(np.argsort(horz_odd_sites)), 
        list(np.argsort(vert_sites))]
    # perm_set = [
    #     horz_even_sites, 
    #     horz_odd_sites, 
    #     vert_sites]

    Vlist_start = []
    perms = []
    for i, c in zip(method_start.indices, method_start.coeffs):
        Vlist_start.append(scipy.linalg.expm(-1j*c*t*hloc[i]))
        perms.append(perm_set[i])

    # perform optimization
    Vlist, f_iter, err_iter = oc.optimize_brickwall_circuit(2 * L, expiH, Vlist_start, perms, **kwargs)

    # convergence plot
    nsteps_stra = np.array([2**i for i in range(6)])
    err_stra = np.zeros(len(nsteps_stra))
    stra = oc.SplittingMethod.suzuki(3, 1)
    print("stra.coeffs:", stra.coeffs)
    for i, nsteps in enumerate(nsteps_stra):
        print("nsteps:", nsteps)
        dt = t / nsteps
        print("dt:", dt)
        W = trotterized_time_evolution(2*L, hloc, perm_set, stra, dt, nsteps)
        err_stra[i] = np.linalg.norm(W - expiH, ord=2)
        print(f"err_stra[{i}]: {err_stra[i]}")

    dt_list = t / nsteps_stra
    plt.loglog(dt_list, err_stra, '.-', label="Strang")
    plt.loglog(dt_list, 6*np.array(dt_list)**2, '--', label="Δt^2")
    plt.xlabel("Δt")
    plt.ylabel("error")
    plt.legend()
    plt.title(f"real-time evolution up to t = {t} using Strang splitting")
    plt.show()

    # visualize optimization progress
    print(f"err_iter before: {err_iter[0]}")
    print(f"err_iter after {len(err_iter)-1} iterations: {err_iter[-1]}")
    plt.semilogy(range(len(err_iter)), err_iter)
    plt.xlabel("iteration")
    plt.ylabel("spectral norm error")
    plt.title(f"optimization progress for a quantum circuit with {len(Vlist)} layers")
    plt.show()
    # rescaled and shifted target function
    # print(f_iter)
    plt.semilogy(range(len(f_iter)), 1 + (np.array(f_iter)) / 2**(L * 2))
    plt.xlabel("iteration")
    plt.ylabel(r"$1 + f(\mathrm{Vlist})/2^L$")
    plt.title(f"optimization target function for a quantum circuit with {len(Vlist)} layers")
    plt.show()

    # save results to disk
    f_iter = np.array(f_iter)
    err_iter = np.array(err_iter)
    with h5py.File(f"ising1d_dynamics_opt_n{nlayers}.hdf5", "w") as f:
        f.create_dataset("Vlist", data=Vlist)
        f.create_dataset("f_iter", data=f_iter)
        f.create_dataset("err_iter", data=err_iter)
        # store parameters
        f.attrs["L"] = L
        f.attrs["J"] = float(J)
        f.attrs["U"] = float(U)
        f.attrs["t"] = float(t)

def main():

    # 3 layers
    # TODO: more layers, other splitting methods
    # use a single Strang splitting step as starting point for optimization
    strang = oc.SplittingMethod.suzuki(3, 1)
    indices_start_n9, coeffs_start_n9 = oc.merge_layers(2*strang.indices, 2*strang.coeffs)
    # divide by 2 since we are taking two steps
    coeffs_start_n9 = [0.5*c for c in coeffs_start_n9]
    print("coeffs_start_n9:", coeffs_start_n9)
    print("indices_start_n9:", indices_start_n9)
    method_start_n9 = oc.SplittingMethod(3, indices_start_n9, coeffs_start_n9, 2)
    # TODO: oc.SplittingMethod.yoshida4
    # strang = oc.SplittingMethod.yoshida4()
    # print(len(strang.coeffs))
    construct_fermi_hubbard_dynamics_opt(9, method_start_n9, niter=10)

if __name__ == "__main__":
    main()