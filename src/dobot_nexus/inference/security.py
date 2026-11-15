"""Shared serving authentication; secrets are supplied through the environment."""
import hmac
import ipaddress
import os


def serving_token(host=None, token=None):
    token = os.environ.get('NEXUS_POLICY_TOKEN', '') if token is None else token
    if token and len(token.encode('utf-8')) < 32:
        raise ValueError('NEXUS_POLICY_TOKEN must contain at least 32 bytes.')
    if host is not None:
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host.lower() == 'localhost'
        if not loopback and not token:
            raise ValueError('A non-loopback policy bind requires NEXUS_POLICY_TOKEN (32+ bytes).')
    return token


def authorized(header, token):
    return not token or hmac.compare_digest((header or '').encode(), ('Bearer ' + token).encode())
