"""
Here we implement the necessary functions to build the circuit for Quantum Gaussian Process Quadrature. 
using the Hilbert Space approximation of the kernel.
"""

import numpy as np
import Bayessian_quadratures
import scipy
from operator import itemgetter

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.circuit.library import QFT
from qiskit.circuit.library.standard_gates import RYGate
from qiskit.circuit.library.data_preparation.state_preparation import StatePreparation
from qiskit.extensions import HamiltonianGate
from qiskit.quantum_info import Statevector as st

backend = Aer.get_backend("aer_simulator")

def compare_binary_strings(bin1, bin2):
    """
    This function compares two binary strings and returns the number of equal 1's.
    """

    if len(bin1) != len(bin2):
        raise ValueError("The binary strings have different length.")
    
    ## transform the binary strings into decimal numbers
    dec1 = int(bin1,2)/2**len(bin1)
    dec2 = int(bin2,2)/2**len(bin2)

    ## avoid problems with the case when both binary strings are 0
    if (dec1 == 0) and (dec2 == 0):
        similarity = 1
    else:
        similarity = 1 - abs(dec1 - dec2)/max([dec1, dec2])
    
    return similarity

## Build a function that compares two binary strings and returns the number of equal 1's
def clean_binary(list_bin):
    """
    This function cleans the binary strings that are repeated in the list.
    """
    
    treshold = 0.9

    for i in range(0, len(list_bin)):
        for j in range(i+1, len(list_bin)):
            ## continue if one of the binary strings is 0
            if (list_bin[i] == str(0)*len(list_bin[0])) or (list_bin[j] == str(0)*len(list_bin[0])):
                continue
            if (compare_binary_strings(list_bin[i], list_bin[j]) >= treshold):
                ## delete list item if the similarity is bigger than the treshold
                list_bin[j] = str(0)*len(list_bin[0])


    list_bin = [i for i in list_bin if i != str("0")*len(list_bin[0])]

    return list_bin

## Function to compute the main counts
def main_counts(counts, n_eig):
    """
    This function returns the counts of the counts before the |0>^n_eig state
    is measured.
    """
    ## save the counts before str(0)*n_eig in a dictionary
    counts_main = {}
    for k in counts.keys():
        if k[:n_eig] != str(0)*n_eig:
            counts_main[k] = counts[k]
        else:
            break
    
    return counts_main

def quantum_eigenvals(X, M, n_eig, delta, shots):

    """
    This function computes the eigenvalues of the matrix Z.T @ Z using quantum phase estimation algorithm.
    In this case, the matrix Z.T @ Z is calculated using the Hilbert Space approximation of the kernel.
    """    
    psi = X.flatten()
    n_psi = int(np.log2(len(psi)))
    T = 2*np.pi/delta

    XXd = X.T @ X 
    psi_vector = st(psi)
    psi_gate = StatePreparation(psi_vector)

    ## Create the circuit
    
    

    ## Creation of quantum registers
    psi_1 = QuantumRegister(n_psi, "\psi")
    eigen = QuantumRegister(n_eig, "\kappa")



    ## Circuit to estimate eigenvalues
    QPE_circuit = QuantumCircuit(eigen,psi_1)

    ## Superposition of the lambda qubits
    for i in range(n_eig):
        QPE_circuit.h(eigen[i])

    ## Evolution operatot for X.T @ X
    U = HamiltonianGate(-XXd, T)
    U.name = "U"

    ## Controlled operations for quantum phase estimation
    for i in range(n_eig):
        Upow = U.power(2**(i)) 
        ctrl_Upow = Upow.control(1)
        C_qubit = eigen[i]
        
        ## Controlled evolution operator over the first M qubits
        QPE_circuit.append(ctrl_Upow, [C_qubit]+psi_1[:int(np.log2(M))])


    ## Quantum fourier transform 
    qfti_gate = QFT(num_qubits= n_eig, inverse=True).to_gate()
    QPE_circuit.append(qfti_gate, eigen)
    

    ## Circuit to measure the eigenvalues
    ## Creation of classical registers
    cr = ClassicalRegister(n_eig,'c')

    ## Circuit intialization
    circuit_measure = QuantumCircuit(eigen, psi_1, cr)
    circuit_measure.append(psi_gate, psi_1)
    
    ## QPE circuit
    circuit_measure.append(QPE_circuit, eigen[:]+psi_1[:])
    circuit_measure.barrier()
        

    for i in range(n_eig):
        circuit_measure.measure(eigen[i],cr[i])

    job = execute(circuit_measure, backend, shots=shots,  seed_simulator=12)
    result = job.result()
    counts = result.get_counts(circuit_measure)

    sort_counts= dict(sorted(counts.items(), key=itemgetter(1), reverse=True))
    
    ## main counts takes exited states that have larger counts than the |0>^n_eig state
    counts_main = main_counts(sort_counts, n_eig)

    theta_bin=[*counts_main.keys()]
    theta_bin = clean_binary(theta_bin)

    if len(theta_bin) > M:
        theta_bin = theta_bin[:M]

    ## if lenght of theta_bin smaller than M, add the smallest possible string to the list
    ## this fix the problem of the last eigenvalue being too small so n_eig bits are not enough to represent it
    if len(theta_bin) < M:
        theta_bin.append(str(0)*(n_eig-1) + str(1))

    eigenvals_quantum = []
    for j in range(0,len(theta_bin)):
        
        zeta=(int(theta_bin[j],2)/(2**n_eig))*(2*np.pi/T)
        
        eigenvals_quantum.append(zeta)

    eigenvals_quantum = np.array(eigenvals_quantum)
        
    ## create a dictionary with the binary strings as key and the eigenvalues as values
    eigenvals_dict = dict(zip(theta_bin, eigenvals_quantum))

    ## sort the dictionary by the eigenvalues in ascending order
    eigenvals_dict = dict(sorted(eigenvals_dict.items(), key=itemgetter(1), reverse=True))

    return eigenvals_dict, QPE_circuit

## Function to compute the mean of the posterior distribution
def mean_QGPR_approximation_posterior(X_mu, Y, Lambda, X, sigma2, q_eigen_dict, QPE_circuit, n_eig, R, shots=2**5):
    """
    This function computes the mean of the posterior distribution of f evaluated at each of the points in Xp conditioned on (X, y)
    using the quantum Gaussian Process Regression with the Hilbert Space approximation of the kernel.
    """
    backend = Aer.get_backend("aer_simulator")

    ## Mean calculation
    psi_1 = X.flatten()
    psi_1_vector = st(psi_1)
    psi_1_gate = StatePreparation(psi_1_vector)

    ## Create the circuit
    n_psi = int(np.log2(len(psi_1)))


    ## Creation of quantum registers
    ancilla = QuantumRegister(1, "a")
    psi_1_qr = QuantumRegister(n_psi, "\psi")
    eigen_qr = QuantumRegister(n_eig, "\kappa")

    qc_psi_1 = QuantumCircuit(ancilla, eigen_qr, psi_1_qr)

    ## initialize the psi register in the state psi
    qc_psi_1.append(psi_1_gate, psi_1_qr)

    ## Apply QPE_ciruit to the registers psi_1 and eigen
    qc_psi_1.append(QPE_circuit, eigen_qr[:] + psi_1_qr[:])

    ## unpack the eigenvalues and the binary strings from the dictionary
    theta_bin = [*q_eigen_dict.keys()]
    quantum_eigvals = [*q_eigen_dict.values()]

    C = quantum_eigvals[R-1] + sigma2 

    for j in range(0,R):
        binary_string =theta_bin[j]
        count_ones = binary_string.count("1")

        zeta=quantum_eigvals[j]

        one_positions = [i for i, digit in enumerate(binary_string[::-1]) if digit == "1"]

        cry=RYGate(2*np.arcsin(C/(zeta + sigma2))).control(count_ones)
        #print("M rotation angle: ", 2*np.arcsin(C/(zeta + sigma2)))
        control_qubits = [eigen_qr[i] for i in one_positions]
        target_qubit = ancilla
        qc_psi_1.append(cry, control_qubits+[target_qubit] )

    qc_psi_1.append(QPE_circuit.inverse(), eigen_qr[:] + psi_1_qr[:])


    Hadamard_circuit1 = qc_psi_1.decompose().to_gate().control(1, ctrl_state='0')

    ## circuit for the estimation in the new point X_*
    Phi2=np.kron(Y, X_mu)
    Phi2_norm=np.linalg.norm(Phi2)
    Phi2=Phi2/Phi2_norm
    PSI2 = Phi2.flatten()

    phis_y_vector=st(PSI2)
    phis_y_gate = StatePreparation(phis_y_vector)

    ancilla_2 = QuantumRegister(1, "a")
    eigen_2_qr = QuantumRegister(n_eig, "\kappa")
    psi_2_qr = QuantumRegister(n_psi, "\psi")

    qc_2 = QuantumCircuit(ancilla_2,eigen_2_qr, psi_2_qr)

    qc_2.append(phis_y_gate, psi_2_qr)
    qc_2.x(ancilla_2)

    Hadamard_circuit2 = qc_2.decompose().to_gate().control(1, ctrl_state='1')

    ## hadamard test
    psi_h = QuantumRegister(n_psi, "\psi")
    eigen_h = QuantumRegister(n_eig, "\kappa")
    ancilla_h = QuantumRegister(1, "a")
    ancilla_h2 = QuantumRegister(1, "a2")

    cr = ClassicalRegister(1, "c")

    qc_h = QuantumCircuit(ancilla_h, eigen_h, psi_h, ancilla_h2, cr)

    qc_h.h(ancilla_h2)
    qc_h.append(Hadamard_circuit1, ancilla_h2[:] + ancilla_h[:] + eigen_h[:] + psi_h[:])
    qc_h.append(Hadamard_circuit2, ancilla_h2[:] + ancilla_h[:] + eigen_h[:] + psi_h[:])
    qc_h.h(ancilla_h2)

    qc_h.measure(ancilla_h2, cr)

    backend = Aer.get_backend("aer_simulator")
    job_ht = execute(qc_h, backend, shots=shots)
    result_ht = job_ht.result()
    counts_ht = result_ht.get_counts(qc_h)

    Exp = 0
    prob_0 = counts_ht['0']/shots
    Exp = 2*prob_0 - 1
    mean_sim = (Exp/C)*Phi2_norm

    return mean_sim

def var_QGPR_approximation_posterior(X_mu, Lambda, X, sigma2, q_eigen_dict, QPE_circuit, n_eig, R, M, shots=2**14):
    
    theta_bin = [*q_eigen_dict.keys()]
    quantum_eigvals = [*q_eigen_dict.values()]
    #print("psi_x:", np.round(X, 5))
    ## Mean calculation
    psi_1 = X.flatten()
    psi_1_vector = st(psi_1)
    psi_1_gate = StatePreparation(psi_1_vector)

    ## Create the circuit
    n_psi = int(np.log2(len(psi_1)))

    ## Creation of quantum registers
    ancilla_r = QuantumRegister(1, "a_r")
    ancilla_s = QuantumRegister(1, "a_s")
    psi_1_qr = QuantumRegister(n_psi, "\psi_1")
    eigen_qr = QuantumRegister(n_eig, "\kappa")

    cr_r = ClassicalRegister(1, "c_r")
    cr_s = ClassicalRegister(1, "c_s")
    

    Phi2=X_mu
    Phi2_norm=np.linalg.norm(Phi2)
    Phi2=Phi2/Phi2_norm
    PSI2 = Phi2.flatten()

    phi_mu_vector=st(PSI2)
    phi_mu_gate = StatePreparation(phi_mu_vector)

    n_psi_2 = int(np.log2(len(PSI2)))
    psi_2_qr = QuantumRegister(n_psi_2, "\psi_2")
    

    qc_swap = QuantumCircuit(ancilla_r, eigen_qr, psi_1_qr, psi_2_qr, ancilla_s, cr_r, cr_s)

    ## initialize the psi register in the state psi
    qc_swap.append(psi_1_gate, psi_1_qr)
    qc_swap.append(phi_mu_gate, psi_2_qr)

    ## Apply QPE_ciruit to the registers psi_1 and eigen
    qc_swap.append(QPE_circuit, eigen_qr[:] + psi_1_qr[:])
    lmn = quantum_eigvals[R-1]
    C_2 = np.sqrt(lmn*(lmn + sigma2))
    alpha=0

    for j in range(0,R):
        binary_string =theta_bin[j]
        count_ones = binary_string.count("1")

        zeta=quantum_eigvals[j]
        #print((1/(zeta+sigma2)))
        alpha=alpha+(1/(zeta+sigma2))

        one_positions = [i for i, digit in enumerate(binary_string[::-1]) if digit == "1"]
        #print("V rotation angle: ", 2*np.arcsin(C/np.sqrt(zeta*(zeta + sigma2))))
        cry=RYGate(2*np.arcsin(C_2/np.sqrt(zeta*(zeta + sigma2)))).control(count_ones)

        control_qubits = [eigen_qr[i] for i in one_positions]
        target_qubit = ancilla_r
        qc_swap.append(cry, control_qubits+[target_qubit] )

    #qc_swap.append(QPE_circuit.inverse(), eigen_qr[:] + psi_1_qr[:])

    qc_swap.h(ancilla_s)
    for i in range(0, n_psi_2):
        qc_swap.cswap(ancilla_s, psi_2_qr[i], psi_1_qr[i])
    qc_swap.h(ancilla_s)

    qc_swap.measure(ancilla_r, cr_r)
    qc_swap.measure(ancilla_s, cr_s) 

    backend = Aer.get_backend("aer_simulator")
    job_st = execute(qc_swap, backend, shots=shots, seed_simulator=1)
    result_st = job_st.result()
    counts_st = result_st.get_counts(qc_swap)
    #print("counts:", counts_st)

    if '1 1' not in counts_st.keys():
        counts_st['1 1'] = 0
    if '1 0' not in counts_st.keys():
        counts_st['1 0'] = 0
    if '0 1' not in counts_st.keys():
        counts_st['0 1'] = 0
    if '0 0' not in counts_st.keys():
        counts_st['0 0'] = 0

    prob_10 = counts_st['1 0']/shots 
    prob_11 = counts_st['1 1']/shots
    prob_01 = counts_st['0 1']/shots
    prob_00 = counts_st['0 0']/shots

    var_11 = abs(alpha - (2*prob_11/C_2**2))*Phi2_norm**2 
    #print("var 11:", var_11)

    var_both =((prob_10 - prob_11)/(C_2*C_2))*Phi2_norm**2
    #print("var both:", var_both)
    ## print probabilities
    #print("alpha:", alpha)
    #print("prob 10 + prob 11:", (prob_10 + prob_11)/(C_2*C_2))
    #print("prob 01 + prob 00:",(1- prob_01 - prob_00)/(C_2*C_2))
    return var_11, qc_swap

def QGPQ_approximation_posterior(mean_args):
    x_eval = mean_args[0]
    y_eval = mean_args[1]
    sigma2 = mean_args[2]
    M = mean_args[3]
    L = mean_args[4]
    alpha = mean_args[5]
    scale = mean_args[6]
    domain = mean_args[7]
    delta = mean_args[8]
    n_eig = mean_args[9]
    R = mean_args[10]
    shots = mean_args[11]

    HSQ= Bayessian_quadratures.Sine_HSQ((x_eval, y_eval),
                    sigma2=sigma2,
                    M = M,
                    L = L,
                    alpha=alpha,
                    scale=scale,
                    domain=domain)

    Phi = HSQ.Phi_matrix(x_eval)
    Lambda = HSQ.Lambda()
    
    X = np.array(Phi @ np.sqrt(Lambda))

    ## Normalize the data
    norm_X = np.linalg.norm(X)

    X_norm = X/norm_X

    XTX = np.array(X_norm.T @ X_norm)
    real_eigenvals, real_eigenvecs = scipy.linalg.eig(XTX)
    real_eigenvals = np.sort(real_eigenvals)[::-1]
    print("Real eigenvalues bef: ", real_eigenvals)
    X_shape = X.shape
    if np.log2(X_shape[0])%1 or np.log2(X_shape[1])%1:
        n_0 = 2**int(np.ceil(np.log2(X_shape[0])))
        n_1 = 2**int(np.ceil(np.log2(X_shape[1])))
        X = np.pad(X, ((0, n_0-X_shape[0]), (0, n_1-X_shape[1])), 'constant')

    new_n = X.shape[0]
    new_m = X.shape[1]
    ## if y_eval is not a power of 2, we add zeros to the vector
    if np.log2(len(y_eval)) % 1 != 0:
        y_eval = np.append(y_eval, np.zeros(2**int(np.log2(len(y_eval))+1) - len(y_eval)))

    ## Normalize the data
    norm_X = np.linalg.norm(X)

    X_norm = X/norm_X

    XTX = np.array(X_norm.T @ X_norm)

    ## For demonstration, we calculate the eigenvalues of XXd and chose delta based on that
    real_eigenvals, real_eigenvecs = scipy.linalg.eig(XTX)
    real_eigenvals = np.sort(real_eigenvals)[::-1]
    
    ## delta parameter should be 1>delta>lam_max
    q_eigen_dict, QPE_circuit = quantum_eigenvals(X_norm, new_m, n_eig, delta, shots)

    ## printing the real and quantum eigenvalues to verify that they are similar
    print("Real eigenvalues after: ", real_eigenvals)
    print("Quantum eigenvalues: ", [*q_eigen_dict.values()])
    print("Quantum eigenvalues keys: ", [*q_eigen_dict.keys()])
    ## Parameters
    mu = 0
    v = 0
    Phi_mu = HSQ.phi_mu_vector()
    X_mu =np.array(Phi_mu.T @ np.sqrt(Lambda)) 
    if np.log2(len(X_mu))%1:
        n_0 = 2**int(np.ceil(np.log2(len(X_mu))))
        X_mu = np.pad(X_mu, (0, n_0-len(X_mu)), 'constant')

    mu = mean_QGPR_approximation_posterior(X_mu, y_eval, Lambda, X_norm, sigma2, q_eigen_dict, QPE_circuit, n_eig, R,  shots)

    v, qc_swap = var_QGPR_approximation_posterior(X_mu, Lambda, X_norm, sigma2, q_eigen_dict, QPE_circuit, n_eig, R, M, shots)
    QBQ_mean = mu/norm_X
    QBQ_var = sigma2*v/norm_X**2
    return QBQ_mean, QBQ_var, qc_swap
