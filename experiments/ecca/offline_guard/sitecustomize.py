"""Disable remote sockets, including implicit model downloads; allow local DDP."""
import socket
import ipaddress
import urllib.request
from urllib.parse import urlparse
try:
    _self_ips = set(socket.gethostbyname_ex(socket.gethostname())[2])
except OSError:
    _self_ips = set()
_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex


def _local(address):
    if not isinstance(address, tuple):
        return True  # UNIX-domain sockets
    host = str(address[0]).lower()
    if host in {'localhost', socket.gethostname().lower()} or host in _self_ips:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _connect(self, address):
    if not _local(address):
        raise OSError('MNS offline mode: remote network is disabled; use existing local weights/caches')
    return _original_connect(self, address)


def _connect_ex(self, address):
    if not _local(address):
        raise OSError('MNS offline mode: remote network is disabled')
    return _original_connect_ex(self, address)

socket.socket.connect = _connect
socket.socket.connect_ex = _connect_ex

_urlopen = urllib.request.urlopen

def _offline_urlopen(url, *args, **kwargs):
    address = url.full_url if hasattr(url, 'full_url') else str(url)
    parsed = urlparse(address)
    if parsed.scheme not in ('', 'file') and not _local((parsed.hostname or '', parsed.port or 443)):
        raise OSError('MNS offline mode: remote URL resource access is disabled')
    return _urlopen(url, *args, **kwargs)

urllib.request.urlopen = _offline_urlopen
