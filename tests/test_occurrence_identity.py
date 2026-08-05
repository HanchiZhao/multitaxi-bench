def test_occurrence_identity_contract():
    ids=[f'a__scenario_1@{i:03d}' for i in range(5)]
    assert len(ids)==len(set(ids))
    assert 'a__scenario_1@001'!='b__scenario_1@001'
