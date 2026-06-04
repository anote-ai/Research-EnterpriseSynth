import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from enterprisesynth.core import SchemaParser
from enterprisesynth.data import crm_spec, finance_spec, devops_spec


def test_crm_spec_has_paths():
    spec = crm_spec()
    assert "paths" in spec
    assert len(spec["paths"]) == 3


def test_finance_spec_parses_to_3_endpoints():
    schema = SchemaParser().parse_openapi(finance_spec())
    assert len(schema.endpoints) == 3


def test_devops_spec_parses_to_3_endpoints():
    schema = SchemaParser().parse_openapi(devops_spec())
    assert len(schema.endpoints) == 3


def test_all_specs_have_info_key():
    for spec in [crm_spec(), finance_spec(), devops_spec()]:
        assert "info" in spec
