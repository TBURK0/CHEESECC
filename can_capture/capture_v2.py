#!/usr/bin/env python3
"""Receive CAN_LOGGER_V2 output without sending serial commands."""
import argparse
import csv
import time


def parse_frame(line):
    parts = line.split(',')
    if len(parts) != 6:
        raise ValueError('expected six fields')
    stamp, ident, ext, rtr, dlc, payload = parts
    stamp, ident = int(stamp), int(ident, 16)
    ext, rtr, dlc = int(ext), int(rtr), int(dlc)
    if not (0 <= stamp <= 0xFFFFFFFF and ext in (0, 1)
            and rtr in (0, 1) and 0 <= dlc <= 8):
        raise ValueError('invalid fields')
    if not 0 <= ident <= (0x1FFFFFFF if ext else 0x7FF):
        raise ValueError('invalid CAN ID')
    tokens = payload.split()
    if any(len(t) != 2 for t in tokens):
        raise ValueError('invalid byte width')
    data = bytes(int(t, 16) for t in tokens)
    if len(data) != (0 if rtr else dlc):
        raise ValueError('DLC mismatch')
    return [stamp, '0x{:X}'.format(ident), ext, rtr, dlc,
            ' '.join('{:02X}'.format(b) for b in data)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('port', nargs='?', help='e.g. COM5')
    parser.add_argument('output', nargs='?', help='new CSV filename')
    parser.add_argument('--list-ports', action='store_true')
    parser.add_argument('--seconds', type=float, default=0,
                        help='stop automatically; default: until Ctrl+C')
    args = parser.parse_args()
    try:
        import serial
        from serial.tools import list_ports
    except ImportError:
        parser.exit(1, 'Install dependency: python -m pip install pyserial\n')
    if args.list_ports:
        ports = list(list_ports.comports())
        for p in ports:
            print('{}: {}'.format(p.device, p.description))
        if not ports:
            print('No serial ports detected.')
        return
    if not args.port or not args.output or args.seconds < 0:
        parser.error('provide PORT OUTPUT.csv and a nonnegative duration')
    count = bad = previous = 0
    pending = bytearray()
    start = last = time.monotonic()
    try:
        # Exclusive create protects previous captures. Opening serial may reset
        # the board; never discard the input buffer after opening it.
        with open(args.output, 'x', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['host_time_s', 'device_us', 'id', 'extended',
                             'rtr', 'dlc', 'data'])
            with serial.Serial(args.port, 1000000, timeout=0.1) as ser:
                start = last = time.monotonic()
                print('Recording {} -> {}; Ctrl+C stops.'.format(args.port, args.output))
                try:
                    while True:
                        now = time.monotonic()
                        if args.seconds and now - start >= args.seconds:
                            break
                        chunk = ser.read(min(max(ser.in_waiting, 1), 65536))
                        host_stamp = time.time()
                        pending.extend(chunk)
                        while b'\n' in pending:
                            raw, _, remainder = pending.partition(b'\n')
                            pending = bytearray(remainder)
                            try:
                                line = raw.decode('ascii').strip()
                                if not line:
                                    continue
                                if line.startswith('#'):
                                    print('\n' + line)
                                    if line.startswith('# ERROR'):
                                        raise RuntimeError(line)
                                    continue
                                frame = parse_frame(line)
                            except (ValueError, UnicodeError):
                                bad += 1
                                if bad <= 3:
                                    print('\nIgnored malformed line: {!r}'.format(raw[:120]))
                                continue
                            writer.writerow(['{:.6f}'.format(host_stamp)] + frame)
                            count += 1
                        if len(pending) > 4096:
                            pending.clear()
                            bad += 1
                        now = time.monotonic()
                        if now - last >= 1:
                            f.flush()
                            print('\r{} frames | {:.1f} frames/s | {} malformed'.format(
                                count, (count - previous) / (now - last), bad),
                                end='', flush=True)
                            previous, last = count, now
                except KeyboardInterrupt:
                    pass
                finally:
                    f.flush()
    except (OSError, serial.SerialException, RuntimeError) as exc:
        parser.exit(1, '\nCapture failed: {}\n'.format(exc))
    elapsed = max(time.monotonic() - start, 0.000001)
    print('\nSaved {} frames in {:.1f}s ({:.1f}/s); {} malformed; {} partial bytes.'.format(
        count, elapsed, count / elapsed, bad, len(pending)))


if __name__ == '__main__':
    main()
