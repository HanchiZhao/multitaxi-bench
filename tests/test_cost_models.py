import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from cost_models import CostModel,take_home
def test_owner_cost():
    m=CostModel('owner',1,0.7,0.7); x=take_home(100,10,5,120,m); assert abs(x['take_home']-89.5)<1e-9
def test_fixed_cost():
    m=CostModel('fixed',1,.25,.25,9); x=take_home(100,10,5,120,m); assert abs(x['take_home']-(100-3.75-18))<1e-9
