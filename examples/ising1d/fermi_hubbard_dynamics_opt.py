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

class FermiHubbardHamiltonianMod(AbstractOperator):
    """
    Fermi-Hubbard Hamiltonian with
      - kinetic hopping coefficient `t`
      - potential interaction strength `u`
    on a lattice.
    """
    def __init__(self, field: Field, t: float, u: float, spin=True):
        # parameter checks
        if field.particle_type != ParticleType.FERMION:
            raise ValueError(f"expecting a field with fermionic particle type, but received {field.particle_type}")
        if not isinstance(t, float):
            raise ValueError(f"expecting a float for 't', received {type(t)}")
        if not isinstance(u, float):
            raise ValueError(f"expecting a float for 'u', received {type(u)}")
        if spin:
            if not isinstance(field.lattice, LayeredLattice):
                raise ValueError("expecting a layered lattice when 'spin' is True")
            if field.lattice.nlayers != 2:
                raise ValueError(f"layered lattice must have two layers (instead of {field.lattice.nlayers})")
        self.t = t
        self.u = u
        self.spin = spin
        self.field = field

    def is_unitary(self):
        """
        Whether the Hamiltonian is unitary.
        """
        # unitary only in some non-typical cases,
        # so returning False here for simplicity
        return False

    def is_hermitian(self):
        """
        Whether the Hamiltonian is Hermitian.
        """
        return True

    def as_field_operator(self):
        """
        Represent the Hamiltonian as FieldOperator.
        """
        latt = self.field.lattice
        L = latt.nsites
        adj = latt.adjacency_matrix()
        if self.spin:
            assert L % 2 == 0
            kin_coeffs = -self.t * np.kron(np.identity(2), adj[:(L//2), :(L//2)])
            int_coeffs = np.zeros((L, L, L, L))
            for i in range(L//2):
                int_coeffs[i, i, i + L//2, i + L//2] = self.u
        else:
            kin_coeffs = self.t * adj
            int_coeffs = np.zeros((L, L, L, L))
            for i in range(L):
                for j in range(i + 1, L):
                    if adj[i, j] != 0:
                        int_coeffs[i, i, j, j] = self.u
        # kinetic hopping term
        T = FieldOperatorTerm([IFODesc(self.field, IFOType.FERMI_CREATE),
                               IFODesc(self.field, IFOType.FERMI_ANNIHIL)],
                              kin_coeffs)
        # interaction term
        V = FieldOperatorTerm([IFODesc(self.field, IFOType.FERMI_CREATE),
                               IFODesc(self.field, IFOType.FERMI_ANNIHIL),
                               IFODesc(self.field, IFOType.FERMI_CREATE),
                               IFODesc(self.field, IFOType.FERMI_ANNIHIL)],
                              int_coeffs)
        return FieldOperator([T, V])

    def as_matrix(self):
        """
        Generate the (sparse) matrix representation of the Hamiltonian.
        """
        return self.as_field_operator().as_matrix()

    @property
    def nsites(self) -> int:
        """
        Number of underlying lattice sites.
        """
        return self.field.lattice.nsites

    def fields(self):
        """
        List of fields the Hamiltonian acts on.
        """
        return [self.field]

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
    U = 0.75

    # time
    t = 0.25
    print("t:", t)

    # construct Hamiltonian
    # latt = qib.lattice.LayeredLattice(qib.lattice.IntegerLattice((L,), pbc=False), 2)
    # field = qib.field.Field(qib.field.ParticleType.FERMION, latt)
    # H = FermiHubbardHamiltonianMod(field, J, U, True).as_matrix()
    
    H = np.zeros((2**(2*L), 2**(2*L)))
    n_spin = np.eye(2)
    n_spin[0,0] = 0
    a_i1 = a_i = np.zeros((2,2))
    a_i[1,0] = 1
    a_i1[0,1] = 1
    n_spin_big = np.kron(n_spin, np.kron(np.eye(2 ** (L - 1)), n_spin))
    for i in range (L):
        H += U * np.kron(np.eye(2 ** i), np.kron(n_spin_big, np.eye(2 ** (L - i - 1))))
    for i in range (L - 1):
        H += np.kron(np.eye(2 ** i),np.kron(construct_fermi_hubbard_local_kinetic_term(J), np.eye(2**(2 * L - i - 2))))
    for i in range (L, 2*L - 1):
        H += np.kron(np.eye(2 ** i),np.kron(construct_fermi_hubbard_local_kinetic_term(J), np.eye(2**(2 * L - i - 2))))
    H += -J * np.kron(a_i1, np.kron(np.eye(2**(L - 2)), np.kron(a_i, np.eye(2**(L)))))
    H += -J * np.kron(np.kron(np.eye(2**(L)), a_i1), np.kron(np.eye(2**(L - 2)), a_i))

    print("H.shape:", H.shape)

    # reference global unitary
    expiH = scipy.linalg.expm(-1j*H*t)

    hloc_horz = construct_fermi_hubbard_local_kinetic_term(J)
    hloc_vert = construct_fermi_hubbard_local_interaction_term(U)
    hloc = [hloc_horz, hloc_horz, hloc_vert]
    assert len(method_start.coeffs) == nlayers

    horz_even_sites = None
    horz_odd_sites = np.roll(range(L*2), -1)
    horz_odd_sites[[L - 1, L*2 -1]] = horz_odd_sites[[L*2 - 1, L -1]]
    vert_sites = np.array([[i, i + L] for i in range(L)]).flatten()

    # Vlist_start = [scipy.linalg.expm(-1j*c*t*hloc) for c in coeffs_start]
    # perms = [None if i % 2 == 0 else np.roll(range(2*L), -1) for i in range(len(Vlist_start))]
    perm_set = [horz_even_sites, horz_odd_sites, vert_sites, horz_odd_sites, horz_even_sites]

    Vlist_start = []
    perms = []
    for i, c in zip(method_start.indices, method_start.coeffs):
        Vlist_start.append(scipy.linalg.expm(-1j*c*t*hloc[i]))
        perms.append(perm_set[i])

    # perform optimization
    Vlist, f_iter, err_iter = oc.optimize_brickwall_circuit(2 * L, expiH, Vlist_start, perms, **kwargs)

    # visualize optimization progress
    print(f"err_iter before: {err_iter[0]}")
    print(f"err_iter after {len(err_iter)-1} iterations: {err_iter[-1]}")
    plt.semilogy(range(len(err_iter)), err_iter)
    plt.xlabel("iteration")
    plt.ylabel("spectral norm error")
    plt.title(f"optimization progress for a quantum circuit with {len(Vlist)} layers")
    plt.show()
    # rescaled and shifted target function
    print(f_iter)
    plt.semilogy(range(len(f_iter)), 1 + (np.array(f_iter) + 251.0) / 2**L)
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
    # use a single Strang splitting step as starting point for optimization
    strang = oc.SplittingMethod.suzuki(3, 1)
    construct_fermi_hubbard_dynamics_opt(5, strang, niter=50)

if __name__ == "__main__":
    main()