import pytest
import plaquette_simulator as ps

class TestInitialization:
    
    def test_init(self):
        num_qubits = 3
        batch_size = 10
        seed = 0
        state = ps.CliffordState(num_qubits, batch_size, seed, True)
        num_qubits_check = state.get_num_qubits()
        batch_size_check = state.get_batch_size()
        seed_check = state.get_seed()
        
        assert num_qubits_check == num_qubits
        assert batch_size_check == batch_size
        assert seed_check == seed
