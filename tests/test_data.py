from enterprisesynth.data import crm_spec, finance_spec, devops_spec, healthcare_spec, legal_spec
from enterprisesynth.core import SchemaParser


def test_crm_spec_structure():
    spec = crm_spec()
    assert spec["info"]["title"] == "CRM API"
    assert len(spec["paths"]) == 3


def test_finance_spec_structure():
    spec = finance_spec()
    assert spec["info"]["title"] == "Finance API"
    assert len(spec["paths"]) == 3


def test_devops_spec_structure():
    spec = devops_spec()
    assert spec["info"]["title"] == "DevOps API"
    assert len(spec["paths"]) == 3


def test_healthcare_spec_structure():
    spec = healthcare_spec()
    assert spec["info"]["title"] == "Healthcare EHR API"
    assert len(spec["paths"]) == 4


def test_legal_spec_structure():
    spec = legal_spec()
    assert spec["info"]["title"] == "Legal DMS API"
    assert len(spec["paths"]) == 4


def test_all_specs_parse_cleanly():
    parser = SchemaParser()
    for spec_fn in [crm_spec, finance_spec, devops_spec, healthcare_spec, legal_spec]:
        schema = parser.parse_openapi(spec_fn())
        assert len(schema.endpoints) > 0
