from src.fraud_alert import score_transaction


class FakeModel:
    """A fake model whose predict_proba is fixed, so risk bands can be tested precisely."""

    def __init__(self, fraud_proba: float):
        self.fraud_proba = fraud_proba

    def predict_proba(self, X):
        return [[1 - self.fraud_proba, self.fraud_proba]]


def test_score_transaction_low_risk():
    result = score_transaction(None, FakeModel(0.1))
    assert result["risk_level"] == "LOW"
    assert result["recommended_action"] == "Approve transaction"
    assert result["fraud_probability"] == 0.1


def test_score_transaction_medium_risk():
    result = score_transaction(None, FakeModel(0.5))
    assert result["risk_level"] == "MEDIUM"
    assert result["recommended_action"] == "Flag for manual review"


def test_score_transaction_high_risk():
    result = score_transaction(None, FakeModel(0.9))
    assert result["risk_level"] == "HIGH"
    assert result["recommended_action"] == "Block transaction & alert customer"


def test_score_transaction_respects_custom_thresholds():
    # With a tight low threshold, a 0.2 probability should now be MEDIUM, not LOW
    result = score_transaction(None, FakeModel(0.2), low=0.1, high=0.5)
    assert result["risk_level"] == "MEDIUM"
