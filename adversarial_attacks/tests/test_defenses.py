import pytest

defense_adv = __import__('simulation.defense_adversarial_training', fromlist=['run_adversarial_training_demo'])
defense_rand = __import__('simulation.defense_input_randomization', fromlist=['run_input_randomization_demo'])
monitoring = __import__('simulation.monitoring_example', fromlist=['run_monitoring_demo'])

def test_adversarial_training():
    clean_acc, adv_acc = defense_adv.run_adversarial_training_demo()
    assert clean_acc >= 0.8  # Should be high
    assert adv_acc > 0.5     # Should be better than random guessing
    assert adv_acc <= clean_acc  # Defense should not make adversarial accuracy higher

def test_input_randomization():
    acc = defense_rand.run_input_randomization_demo()
    assert 0 <= acc <= 1

def test_monitoring():
    flags = monitoring.run_monitoring_demo()
    assert 5 in flags  # The injected outlier should be detected
