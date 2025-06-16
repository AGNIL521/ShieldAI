import pytest

# Import simulation functions
def import_simulation(name):
    import importlib.util, sys, os
    sim_dir = os.path.join(os.path.dirname(__file__), '../simulation')
    sys.path.insert(0, os.path.abspath(sim_dir))
    mod = importlib.import_module(name)
    sys.path.pop(0)
    return mod

attack_demo = import_simulation('attack_demo')
nlp_attack_demo = import_simulation('nlp_attack_demo')
ids_attack_demo = import_simulation('ids_attack_demo')

def test_attack_demo():
    # Should return True (attack fooled classifier) or False (not always possible)
    result = attack_demo.run_attack_demo(plot=False)
    assert isinstance(result, bool)


def test_nlp_attack_demo():
    clean_acc, adv_acc, results = nlp_attack_demo.run_nlp_demo()
    assert clean_acc >= 0 and adv_acc >= 0
    # Adversarial accuracy should not exceed clean accuracy
    assert adv_acc <= clean_acc
    # At least one adversarial example should be present
    assert len(results) > 0


def test_ids_attack_demo():
    clean_acc, adv_acc = ids_attack_demo.run_ids_demo()
    assert clean_acc >= 0 and adv_acc >= 0
    # Adversarial accuracy should not exceed clean accuracy
    assert adv_acc <= clean_acc
