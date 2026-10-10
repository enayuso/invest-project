# app/domain/exceptions.py
# Broker-agnostic exception hierarchy shared by every broker adapter.
# 브로커에 독립적인 공통 예외 계층 (KIS 전용이 아님)


class BrokerError(Exception):
    """Base for any error raised by a broker adapter (KIS or future ones).
    브로커 어댑터가 발생시키는 모든 에러의 최상위 클래스"""
