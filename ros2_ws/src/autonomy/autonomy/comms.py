"""HMAC-authenticated command envelopes with receiver-owned sessions.

Integrity/authentication only; confidentiality and ROS graph ACLs are separate.
Never use an attacker-selected receiver session or operator identity.
"""
import hashlib
import hmac
import json
import secrets

from .core import CommandInbox


def canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def sign_command(payload, key, session, operator):
    envelope = dict(session=session, operator=operator, command=payload)
    envelope['mac'] = hmac.new(key, canonical(envelope), hashlib.sha256).hexdigest()
    return json.dumps(envelope, separators=(',', ':'), allow_nan=False)


class AuthenticatedInbox:
    def __init__(self, key, operator, session=None):
        if not isinstance(key, bytes) or len(key) < 32:
            raise ValueError('command key must contain at least 32 bytes')
        if not isinstance(operator, str) or not 1 <= len(operator) <= 64:
            raise ValueError('invalid operator identity')
        self.key, self.operator = key, operator
        self.session = session if session is not None else secrets.token_hex(16)
        self.inbox = CommandInbox()

    def receive(self, line, now):
        if not isinstance(line, str) or len(line) > 2048:
            raise ValueError('invalid command envelope')
        try:
            envelope = json.loads(line)
            if not isinstance(envelope, dict) or set(envelope) != {'session', 'operator', 'command', 'mac'}:
                raise ValueError('invalid command envelope')
            mac = envelope.pop('mac')
            if not isinstance(mac, str) or len(mac) != 64 or not mac.isascii():
                raise ValueError('invalid authentication')
            expected = hmac.new(self.key, canonical(envelope), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(mac, expected):
                raise ValueError('invalid authentication')
            if envelope['session'] != self.session or envelope['operator'] != self.operator:
                raise ValueError('unauthorized session/operator')
            return self.inbox.receive(json.dumps(envelope['command'], allow_nan=False), now)
        except (TypeError, OverflowError, RecursionError) as exc:
            raise ValueError('invalid command envelope') from exc


def main():
    """Print a signed envelope for an existing receiver session; never sends/arms."""
    import argparse
    import os
    import time
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--operator', default='driver')
    parser.add_argument('--seq', type=int, required=True)
    parser.add_argument('--stamp', type=float, help='ROS seconds; default wall clock, unsuitable for use_sim_time')
    parser.add_argument('--throttle', type=float, default=0.0)
    parser.add_argument('--steering', type=float, default=0.0)
    parser.add_argument('--brake', type=float, default=0.0)
    args = parser.parse_args()
    try:
        key = bytes.fromhex(os.environ.get('AUTONOMY_COMMAND_KEY', ''))
        AuthenticatedInbox(key, args.operator, args.session)
        stamp = time.time() if args.stamp is None else args.stamp
        payload = dict(seq=args.seq, stamp=stamp, throttle=args.throttle, steering=args.steering, brake=args.brake)
        CommandInbox().receive(json.dumps(payload), stamp)
        print(sign_command(payload, key, args.session, args.operator))
    except ValueError as exc:
        parser.exit(1, 'Command signing failed: %s\n' % exc)


if __name__ == '__main__':
    main()
