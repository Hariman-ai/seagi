"""THE OFFICIAL SCORE (2026-09-11, user: "he needs to play and learn and grow
under the official rules").

Reads the ARC-AGI-3 scorecard the sidecar keeps for his play -- ARC Prize's
own computation: per level (human_baseline / his_actions)^2 * 100 capped at
115, per environment the level scores weighted 1,2,3.. capped at the fraction
of levels completed, MAX over runs, and the card score = mean over the 25
public environments.  A sidecar restart opens a new card and the API scores
nothing into a reused one, so the card alone cannot show growth: this keeps
the BEST RUN PER ENVIRONMENT EVER seen across cards in /root/official_best.json
and reports an all-time official score from those, plus the live card.

Also writes /root/arc_baselines.json: game -> human baseline actions per
level, from the card, so anything in him can read the official budget
(5 x baseline) and target for every level.

Log line (appended to /root/official.log):
  OFFICIAL <utc> card=<id8> card_score=<x> levels=<n>/<N> actions=<n> |
  alltime=<x> envs_scored=<k>/25 levels_ever=<n> | top: g:score(levels) ...
"""
import json, os, socket, sys, time

SOCK = "/run/arc3.sock"
BEST = "/root/official_best.json"
BASE = "/root/arc_baselines.json"
LOG = "/root/official.log"


def card():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(60)
    s.connect(SOCK)
    s.sendall(b'{"cmd": "scorecard"}\n')
    d = json.loads(s.makefile("rb").readline())
    return d.get("id"), d.get("card") or {}


def load(p, default):
    try:
        return json.load(open(p))
    except Exception:
        return default


def main():
    cid, c = card()
    if not isinstance(c, dict) or "environments" not in c:
        line = "OFFICIAL %s card=%s UNAVAILABLE %s" % (
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), str(cid)[:8], json.dumps(c)[:200])
        print(line)
        open(LOG, "a").write(line + "\n")
        return
    best = load(BEST, {})
    bases = load(BASE, {})
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for e in c.get("environments") or []:
        gid = e.get("id") or ""
        g = gid[:4]
        runs = e.get("runs") or []
        if not runs:
            continue
        r = max(runs, key=lambda r: (float(r.get("score") or 0), int(r.get("levels_completed") or 0)))
        base = r.get("level_baseline_actions")
        if base and all(isinstance(b, int) and b > 0 for b in base):
            bases[g] = base
        rec = {"score": float(r.get("score") or 0), "levels": int(r.get("levels_completed") or 0),
               "level_actions": r.get("level_actions"), "baselines": base,
               "card": str(cid), "seen": now, "runs_on_card": len(runs)}
        old = best.get(g)
        if old is None or (rec["score"], rec["levels"]) > (float(old.get("score") or 0), int(old.get("levels") or 0)):
            rec["first_seen"] = (old or {}).get("first_seen") or now
            rec["improved"] = now
            best[g] = rec
        else:
            old["seen"] = now
            old["runs_on_card"] = len(runs)
    json.dump(best, open(BEST, "w"), indent=1, sort_keys=True)
    json.dump(bases, open(BASE, "w"), indent=1, sort_keys=True)
    n_env = int(c.get("total_environments") or 25) or 25
    alltime = sum(float(v.get("score") or 0) for v in best.values()) / max(n_env, len(best))
    scored = sum(1 for v in best.values() if float(v.get("score") or 0) > 0)
    levels_ever = sum(int(v.get("levels") or 0) for v in best.values())
    top = sorted(best.items(), key=lambda kv: -float(kv[1].get("score") or 0))[:6]
    line = ("OFFICIAL %s card=%s card_score=%.3f levels=%s/%s actions=%s | alltime=%.3f envs_scored=%d/%d "
            "levels_ever=%d | top: %s" % (
                now, str(cid)[:8], float(c.get("score") or 0), c.get("total_levels_completed"),
                c.get("total_levels"), c.get("total_actions"), alltime, scored, n_env, levels_ever,
                " ".join("%s:%.1f(%d)" % (g, float(v["score"]), int(v["levels"])) for g, v in top)))
    print(line)
    open(LOG, "a").write(line + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        line = "OFFICIAL %s ERROR %s: %s" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), type(e).__name__, e)
        print(line)
        open(LOG, "a").write(line + "\n")
