"""peak_keeper: every 10 min, score the live policy on the rolling window of
deterministic (suffix_action_mode='mean') full-match episodes.  Whenever that
mean sets a new high, snapshot latest.pt into the peak vault so an overnight
regression cannot destroy the best policy the run ever reached.

Keyed on the deterministic window, never on exploration score_mean -- the
exploration metric was shown to be stddev noise (see the reward-first
fine-tune post-mortem).
"""
import json, os, glob, shutil, time

VAULT = os.environ.get("PEAK_VAULT", "/root/preserved/auto_peaks")
LOG = os.environ.get("PEAK_LOG", "/root/peak_keeper.log")
WINDOW = 40        # episodes in the rolling deterministic window
MIN_N = 25         # do not judge on a thin window
FLOOR = float(os.environ.get("PEAK_FLOOR", "86.0"))       # must beat the resume checkpoint (BEST_145155 ~ 85.5) first
KEEP = 6           # cap the vault


def log(msg):
    with open(LOG, "a") as f:
        f.write("%s %s\n" % (time.strftime("%F %T"), msg))


def live_run():
    cands = glob.glob("/root/autodl-tmp/runs/stage_blue2_*") + \
            glob.glob("/root/autodl-tmp/runs/stage_red2_*")
    cands = [d for d in cands if os.path.exists(d + "/cycle_telemetry.jsonl")]
    if not cands:
        return None
    return max(cands, key=lambda d: os.path.getmtime(d + "/cycle_telemetry.jsonl"))


def det_window(out):
    rows = []
    with open(out + "/cycle_telemetry.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("reset_mode") == "full" and r.get("suffix_action_mode") == "mean":
                rows.append(r)
    return rows[-WINDOW:]


def prune():
    files = glob.glob(VAULT + "/peak_*.pt")
    if len(files) <= KEEP:
        return
    def score(p):
        try:
            return float(os.path.basename(p).split("_")[1])
        except Exception:
            return 0.0
    for p in sorted(files, key=score)[:len(files) - KEEP]:
        try:
            os.remove(p)
            log("pruned %s" % os.path.basename(p))
        except Exception:
            pass


def main():
    os.makedirs(VAULT, exist_ok=True)
    best = FLOOR
    # resume the high-water mark across restarts
    for p in glob.glob(VAULT + "/peak_*.pt"):
        try:
            best = max(best, float(os.path.basename(p).split("_")[1]))
        except Exception:
            pass
    log("start best=%.1f" % best)
    while True:
        try:
            out = live_run()
            if out and os.path.exists(out + "/latest.pt"):
                rows = det_window(out)
                if len(rows) >= MIN_N:
                    mean = sum((r.get("scored", 0) or 0) for r in rows) / len(rows)
                    if mean > best + 0.5:
                        upd = "?"
                        mp = out + "/metrics.jsonl"
                        if os.path.exists(mp):
                            tail = [l for l in open(mp) if l.strip()]
                            if tail:
                                upd = json.loads(tail[-1]).get("updates", "?")
                        dst = "%s/peak_%.1f_%s_%s.pt" % (
                            VAULT, mean, upd, time.strftime("%H%M"))
                        tmp = dst + ".tmp"
                        shutil.copy2(out + "/latest.pt", tmp)
                        os.replace(tmp, dst)
                        log("PEAK mean=%.1f n=%d upd=%s -> %s"
                            % (mean, len(rows), upd, os.path.basename(dst)))
                        best = mean
                        prune()
        except Exception as exc:
            log("error %r" % (exc,))
        time.sleep(600)


main()
