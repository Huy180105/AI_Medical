import pytest
from fastapi.testclient import TestClient
from src.privacy.homomorphic_engine import HomomorphicEncryptionEngine
from src.api.app import create_app


def test_homomorphic_engine():
    engine = HomomorphicEncryptionEngine()
    features = [3.0, 1.0, 2.0, 0.0, 0.0, 0.85, 80.0, 120.0, 98.0, 37.0, 5.0, 7.5]

    # Context generation
    client_ctx, pub_bytes, sec_bytes = engine.create_context()
    assert len(pub_bytes) > 0
    assert len(sec_bytes) > 0

    # Encryption
    enc_bytes = engine.encrypt_vector(features, client_ctx)
    assert len(enc_bytes) > 0

    # Server evaluation on ciphertext
    enc_outs = engine.evaluate_encrypted(enc_bytes, pub_bytes)
    assert len(enc_outs) == 4

    # Client decryption
    decrypted = engine.decrypt_predictions(enc_outs, client_ctx)
    assert "predicted_risk_level" in decrypted
    assert "probabilities" in decrypted
    assert decrypted["predicted_risk_level"] in ["Low", "Medium", "High", "Emergency"]

    # Benchmark run
    bench = engine.run_benchmark(features)
    assert bench["status"] == "success"
    assert "benchmarks" in bench
    assert bench["precision_accuracy"]["classes_match"] is True


def test_privacy_api_endpoints():
    app = create_app()
    client = TestClient(app)

    # Health
    h_res = client.get("/privacy/fhe/health")
    assert h_res.status_code == 200
    assert h_res.json()["status"] == "healthy"
    assert "CKKS" in h_res.json()["scheme"]

    # Benchmark
    b_res = client.post("/privacy/fhe/benchmark", json={})
    assert b_res.status_code == 200
    b_data = b_res.json()
    assert b_data["status"] == "success"
    assert "benchmarks" in b_data

    # Encrypted predict
    p_res = client.post(
        "/privacy/fhe/predict-encrypted",
        json={"features": [1.0, 0.0, 0.0, 0.0, 0.0, 0.9, 70.0, 115.0, 99.0, 36.6, 2.0, 5.5]}
    )
    assert p_res.status_code == 200
    p_data = p_res.json()
    assert p_data["status"] == "success"
    assert "predicted_risk_level" in p_data["result"]
