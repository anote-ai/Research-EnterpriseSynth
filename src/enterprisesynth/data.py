from __future__ import annotations


def crm_spec() -> dict:
    """Minimal OpenAPI spec for a CRM API with 3 endpoints."""
    return {
        "info": {"title": "CRM API", "version": "1.0.0"},
        "paths": {
            "/contacts": {
                "post": {
                    "operationId": "createContact",
                    "summary": "Create a new contact",
                    "parameters": [
                        {"name": "name", "type": "string", "required": True},
                        {"name": "email", "type": "string", "required": True},
                        {"name": "company", "type": "string", "required": False},
                    ],
                    "responses": {"200": {"schema": {"type": "object"}}},
                }
            },
            "/contacts/{id}": {
                "get": {
                    "operationId": "getContact",
                    "summary": "Get a contact by ID",
                    "parameters": [
                        {"name": "id", "type": "string", "required": True},
                    ],
                    "responses": {"200": {"schema": {"type": "object"}}},
                }
            },
            "/contacts/list": {
                "get": {
                    "operationId": "listContacts",
                    "summary": "List all contacts",
                    "parameters": [
                        {"name": "page", "type": "integer", "required": False},
                        {"name": "limit", "type": "integer", "required": False},
                    ],
                    "responses": {"200": {"schema": {"type": "array"}}},
                }
            },
        },
    }


def finance_spec() -> dict:
    """Minimal OpenAPI spec for a Finance API with 3 endpoints."""
    return {
        "info": {"title": "Finance API", "version": "1.0.0"},
        "paths": {
            "/quote/{ticker}": {
                "get": {
                    "operationId": "getQuote",
                    "summary": "Get real-time stock quote",
                    "parameters": [
                        {"name": "ticker", "type": "string", "required": True},
                    ],
                    "responses": {"200": {"schema": {"type": "object"}}},
                }
            },
            "/account/balance": {
                "get": {
                    "operationId": "getBalance",
                    "summary": "Get account balance",
                    "parameters": [
                        {"name": "account_id", "type": "string", "required": True},
                    ],
                    "responses": {"200": {"schema": {"type": "object"}}},
                }
            },
            "/account/transactions": {
                "get": {
                    "operationId": "getTransactions",
                    "summary": "Get account transactions",
                    "parameters": [
                        {"name": "account_id", "type": "string", "required": True},
                        {"name": "start_date", "type": "string", "required": False},
                        {"name": "end_date", "type": "string", "required": False},
                    ],
                    "responses": {"200": {"schema": {"type": "array"}}},
                }
            },
        },
    }


def devops_spec() -> dict:
    """Minimal OpenAPI spec for a DevOps API with 3 endpoints."""
    return {
        "info": {"title": "DevOps API", "version": "1.0.0"},
        "paths": {
            "/deploy": {
                "post": {
                    "operationId": "deployService",
                    "summary": "Deploy a service to an environment",
                    "parameters": [
                        {"name": "service", "type": "string", "required": True},
                        {"name": "environment", "type": "string", "required": True},
                        {"name": "version", "type": "string", "required": False},
                    ],
                    "responses": {"200": {"schema": {"type": "object"}}},
                }
            },
            "/status/{service}": {
                "get": {
                    "operationId": "getStatus",
                    "summary": "Get deployment status",
                    "parameters": [
                        {"name": "service", "type": "string", "required": True},
                    ],
                    "responses": {"200": {"schema": {"type": "object"}}},
                }
            },
            "/rollback": {
                "post": {
                    "operationId": "rollback",
                    "summary": "Rollback a service to a previous version",
                    "parameters": [
                        {"name": "service", "type": "string", "required": True},
                        {"name": "target_version", "type": "string", "required": True},
                    ],
                    "responses": {"200": {"schema": {"type": "object"}}},
                }
            },
        },
    }
