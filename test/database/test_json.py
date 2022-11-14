import json
import itertools
import pytest
import os

from FisInMa.database import *
from FisInMa.solving import fisher_determinant, fisher_mineigenval, fisher_ratioeigenval, fisher_sumeigenval
from FisInMa.solving import calculate_fisher_criterion
from test.setUp import default_model_small, pool_model_small


def combinations():
    for comb in itertools.product(
        [True, False],
        [fisher_determinant, fisher_mineigenval, fisher_ratioeigenval, fisher_sumeigenval]
    ):
        yield comb


@pytest.mark.parametrize("identical_times,criterion", combinations())
def test_json_default(default_model_small, criterion):
    fsmp = default_model_small.fsmp

    fsr = calculate_fisher_criterion(fsmp, criterion=criterion)

    out_str = json_dumps(fsr)
    json_dump(fsr, "test/outfile.json")
    json_dict_1 = json.loads(out_str)
    fp = open("test/outfile.json")
    json_dict_2 = json.load(fp)

    assert json_dict_1 == json_dict_2
    os.remove("test/outfile.json")


@pytest.mark.parametrize("identical_times,criterion", combinations())
def test_json_pool(pool_model_small, criterion):
    fsmp = pool_model_small.fsmp

    fsr = calculate_fisher_criterion(fsmp, criterion=criterion)

    out_str = json_dumps(fsr)
    json_dump(fsr, "test/outfile.json")
    json_dict_1 = json.loads(out_str)
    fp = open("test/outfile.json")
    json_dict_2 = json.load(fp)

    assert json_dict_1 == json_dict_2
    os.remove("test/outfile.json")
