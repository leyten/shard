#!/usr/bin/env bash
# safe_kill PATTERN [-s SIGNAL]
#
# Kill every process whose FULL COMMAND LINE matches PATTERN (an extended regex, exactly like
# `pgrep -f`) — EXCEPT this script, its shell, the ssh/sshd session that launched it, and every
# ancestor up to init. It can NEVER terminate the command that invoked it.
#
# WHY THIS EXISTS: raw `pkill -f PATTERN` (and `pgrep -f PATTERN | xargs kill`) self-terminate
# whenever the launching command line itself contains PATTERN. The classic footgun is the
# kill-then-relaunch one-liner:
#
#     ssh box "pkill -f m25_lever_bench.py; python m25_lever_bench.py ..."   # <-- kills its OWN shell
#
# `pkill -f` matches the ssh/bash process running that very string (its argv contains the pattern),
# so the shell dies mid-command and the relaunch never happens — a silent, repeated launch-wipe.
# `safe_kill` walks its own ancestor chain and excludes it, so the caller always survives.
#
# For a long-lived DAEMON prefer `pkill -x <exact-process-name>` (matches the argv[0] name, not the
# whole cmdline) — also self-match-proof. Use safe_kill when you must match on the full command line
# (script name, flags, args) and can't rely on a unique process name.
#
# Always exits 0 (nothing-to-kill is success). Prints how many it signalled.
set -u

sig="KILL"
pat=""
while [ $# -gt 0 ]; do
    case "$1" in
        -s|--signal) sig="${2:?-s needs a signal}"; shift 2 ;;
        --) shift; pat="${1:-}"; break ;;
        -*) echo "safe_kill: unknown option $1" >&2; exit 2 ;;
        *) pat="$1"; shift ;;
    esac
done
[ -n "$pat" ] || { echo "usage: safe_kill PATTERN [-s SIGNAL]" >&2; exit 2; }

# Exclusion set = this PID + the full ancestor chain (shell -> ssh -> sshd -> ... -> init).
# Anything in here is us-or-our-parent and must never be killed.
excl=" "
p=$$
while [ -n "$p" ] && [ "$p" -gt 0 ] 2>/dev/null; do
    excl="${excl}${p} "
    np=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    { [ -z "$np" ] || [ "$np" = "$p" ]; } && break
    p="$np"
done

killed=0
for pid in $(pgrep -f -- "$pat" 2>/dev/null); do
    case "$excl" in
        *" ${pid} "*) continue ;;                 # self or an ancestor — never kill
    esac
    if kill -"$sig" "$pid" 2>/dev/null; then
        killed=$((killed + 1))
    fi
done
echo "safe_kill: sent SIG${sig} to ${killed} proc(s) matching '${pat}' (self+ancestors excluded)"
exit 0
