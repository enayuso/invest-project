# app/brokers/base/__init__.py
# Broker-agnostic HTTP plumbing, for any future broker adapter to inherit.
# 향후 추가될 증권사 어댑터가 공통으로 상속할 HTTP 기반 로직

from app.brokers.base.http_client import BaseHttpClient

__all__ = ["BaseHttpClient"]
