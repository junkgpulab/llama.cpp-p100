#!/usr/bin/env python3
"""Real-world decode speed as the context fills, through llama-server with the production flags.

Three modes, all with qwen-server's GPU flags:

  staircase (default)  one growing conversation: each step sends a longer prefix of the same
                       document plus a question, and the server restores its nearest context
                       checkpoint instead of re-prefilling. Slow (one full prefill), no setup.

  --fill DIR           prefill the document once and save a slot snapshot at each depth into DIR
                       (target and MTP draft state; several GB each, so put DIR on /mnt/fast).

  --restore DIR        for each snapshot in DIR: restore it (seconds), ask a question, measure.
                       The snapshot prompt is an exact token prefix of the question prompt, so
                       only the question is processed. A hybrid model cannot rewind even one
                       token, which is why the prefix is verified token by token when filling.

At each depth it records generation speed, draft acceptance, and prefill of the new tokens. A
watchdog kills the server (by PID) if GPU0 free memory drops under a floor, because GPU0 also
carries the desktop.

Usage: depth-bench.py <build bin dir> <out.jsonl> [--depths 2000,...] [--fill DIR | --restore DIR]
"""
import argparse, json, os, signal, subprocess, sys, threading, time, urllib.request

MODEL = "/mnt/fast/models/Qwen3.8-27B-Q6_K.gguf"
CORPUS = "/mnt/fast/p100-scratch/deep-corpus.txt"
PORT = 8099
FLOOR_MIB = 250

# the user's serving flags (vision on, -ub 2048), minus --host/--tools/--mcp-servers-config
SERVER_ARGS = [
    "-m", MODEL, "-ngl", "99", "-sm", "tensor", "-fa", "1", "-ctk", "q4_0", "-ctv", "q4_0",
    "-c", "262144", "-b", "32768", "-ub", "2048", "-np", "1",
    "--mmproj", "/mnt/fast/models/mmproj-Qwen3.8-27B-Q8_0.gguf",
    "--spec-type", "draft-mtp", "--spec-draft-n-max", "4", "--spec-draft-p-min", "0.2",
    "-ngld", "99", "-ubd", "64", "-ctkd", "q4_0", "-ctvd", "q4_0",
    "--jinja", "--temp", "1.0", "--top-k", "20", "--top-p", "0.95", "--min-p", "0.0",
    "--host", "127.0.0.1", "--port", str(PORT),
]

QUESTIONS = [
    "Summarize the most important build options described above, and explain when each one matters.",
    "Write a short tutorial, with commands, for running a model with the server described above.",
]


def post(path, body, timeout=7200):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def gpu0_free():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
                                   "-i", "0"], text=True)
    return int(out.strip())


def doc_at(text, toks, d):
    """The document cut at a paragraph boundary at or below d tokens, ending in a blank line so
    the question that follows starts a fresh token."""
    doc = post("/detokenize", {"tokens": toks[:d]})["content"]
    cut = doc.rfind("\n\n")
    return doc[:cut + 2] if cut > 0 else doc + "\n\n"


def render(doc, q):
    """(prefix, full): the chat-templated prompt, and its part up to the end of the document."""
    full = post("/apply-template", {"messages": [{"role": "user", "content": doc + q}]})["prompt"]
    i = full.index(doc) + len(doc)
    return full[:i], full


def record(label, d, qi, r, t0, min_free):
    t = r.get("timings", {})
    return {
        "label": label, "depth": d, "q": qi,
        "n_prompt_total": t.get("cache_n", 0) + t.get("prompt_n", 0),
        "prompt_n": t.get("prompt_n"), "prompt_tps": t.get("prompt_per_second"),
        "gen_n": t.get("predicted_n"), "gen_tps": t.get("predicted_per_second"),
        "draft_n": t.get("draft_n"), "draft_accepted": t.get("draft_n_accepted"),
        "wall_s": round(time.time() - t0, 1), "gpu0_min_free": min_free,
        "t_start": t0, "t_end": time.time(),
        "content": r.get("content", r.get("choices", [{}])[0].get("message", {}).get("content")),
    }


def show(rec):
    acc = (rec["draft_accepted"] or 0) / max(1, rec["draft_n"] or 0)
    cyc = max(1, (rec["gen_n"] or 0) - (rec["draft_accepted"] or 0))
    mspc = (rec["gen_n"] or 0) / max(1e-9, rec["gen_tps"] or 0) * 1000 / cyc
    print(f"depth {rec['n_prompt_total']:>6}  gen {rec['gen_tps']:.2f} t/s  accept {acc:.3f}  {mspc:.1f} ms/cycle  "
          f"prompt {rec['prompt_n']} @ {rec['prompt_tps']:.1f} t/s  gpu0 min free {rec['gpu0_min_free']} MiB",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bindir")
    ap.add_argument("out")
    ap.add_argument("--depths", default="2000,16000,32000,64000,128000,192000,256000")
    ap.add_argument("--n-predict", type=int, default=512)
    ap.add_argument("--label", default="")
    ap.add_argument("--n-max", default="", help="comma list of per-request speculative.n_max values (restore mode)")
    ap.add_argument("--server-n-max", type=int, default=0, help="override --spec-draft-n-max on the server")
    ap.add_argument("--server-p-min", type=float, default=-1, help="override --spec-draft-p-min on the server")
    ap.add_argument("--server-prefix", default="", help="command prefix for the server, e.g. an nsys profile invocation")
    ap.add_argument("--seed-offset", type=int, default=0, help="restore mode: first seed index (to draw new seeds)")
    ap.add_argument("--seeds", type=int, default=1, help="restore mode: repeat each question with this many seeds")
    ap.add_argument("--extra-chars", type=int, default=0,
                    help="restore mode: insert this many characters of corpus text before the question, "
                         "to time prefill at the snapshot's depth (~4100 chars is ~1k tokens)")
    ap.add_argument("--ub", type=int, default=0, help="override the server's -ub (default 2048)")
    ap.add_argument("--mmdev", default="", help="device for the vision encoder, e.g. CUDA1 (server -mmdev)")
    ap.add_argument("--graphs", default="3", help="GGML_CUDA_GRAPHS_PRE_VOLTA for the server (production: 3, single-token graphs only)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fill", metavar="DIR")
    g.add_argument("--restore", metavar="DIR")
    a = ap.parse_args()
    a.depths_given = "--depths" in sys.argv
    depths = [int(x) for x in a.depths.split(",")]
    slot_dir = a.fill or a.restore

    args = list(SERVER_ARGS)
    if a.ub:
        args[args.index("-ub") + 1] = str(a.ub)
    if a.mmdev:
        args += ["-mmdev", a.mmdev]
    if a.server_n_max:
        args[args.index("--spec-draft-n-max") + 1] = str(a.server_n_max)
    if a.server_p_min >= 0:
        args[args.index("--spec-draft-p-min") + 1] = str(a.server_p_min)
    if slot_dir:
        os.makedirs(slot_dir, exist_ok=True)
        args += ["--slot-save-path", slot_dir.rstrip("/") + "/"]
    env = dict(os.environ, LD_LIBRARY_PATH=a.bindir, GGML_CUDA_P2P="1", GGML_CUDA_GRAPHS_PRE_VOLTA=a.graphs)
    # qwen-server's defaults for the sampled MTP draft (OPTLOG 202); the environment overrides them
    env.setdefault("LLAMA_SPEC_SAMPLE_TEMP", "1.0")
    env.setdefault("LLAMA_SPEC_DRAFT_TOPK", "20")
    log = open(a.out + ".server.log", "w")
    srv = subprocess.Popen(a.server_prefix.split() + [os.path.join(a.bindir, "llama-server")] + args, env=env,
                           stdout=log, stderr=subprocess.STDOUT)
    min_free = [10**9]
    stop = threading.Event()

    def watchdog():
        while not stop.is_set():
            try:
                f = gpu0_free()
                min_free[0] = min(min_free[0], f)
                if f < FLOOR_MIB:
                    print(f"WATCHDOG: GPU0 free {f} MiB < {FLOOR_MIB}; killing server {srv.pid}", flush=True)
                    srv.send_signal(signal.SIGKILL)
                    return
            except Exception:
                pass
            time.sleep(2)

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                if srv.poll() is not None:
                    sys.exit("server exited during load; see " + a.out + ".server.log")
                time.sleep(1)

        out = open(a.out, "a")
        manifest_path = os.path.join(slot_dir, "manifest.json") if slot_dir else None

        if a.restore:
            manifest = json.load(open(manifest_path))
            want = set(depths) if a.depths_given else None
            nmaxs = [int(x) for x in a.n_max.split(",")] if a.n_max else [None]
            extra = ""
            if a.extra_chars:
                t = open(CORPUS, encoding="utf-8", errors="ignore").read()[:a.extra_chars]
                extra = t[:t.rfind("\n\n") + 2] if "\n\n" in t else t + "\n\n"
            for ent in manifest:
                if want and not any(abs(ent["n_tokens"] - d) < 4096 for d in want):
                    continue
                for nm, si, (qi, q) in [(nm, si, qq) for nm in nmaxs for si in range(a.seed_offset, a.seed_offset + a.seeds) for qq in enumerate(QUESTIONS)]:
                        post(f"/slots/0?action=restore", {"filename": ent["file"]})
                        full = open(os.path.join(slot_dir, ent["prompt_file"]), encoding="utf-8").read() + extra + q
                        full = render_tail(full, ent)
                        body = {"prompt": full, "n_predict": a.n_predict, "cache_prompt": True, "seed": 1234 + qi + 1000 * si}
                        if nm is not None:
                            body["speculative.n_max"] = nm
                        t0 = time.time()
                        r = post("/completion", body)
                        rec = record(a.label + (f" nmax={nm}" if nm is not None else ""), ent["n_tokens"], qi, r, t0, min_free[0])
                        rec["n_max"] = nm
                        rec["seed"] = body["seed"]
                        out.write(json.dumps(rec) + "\n"); out.flush()
                        print(f"n_max={nm} ", end=""); show(rec)
            return

        text = open(CORPUS, encoding="utf-8", errors="ignore").read()
        toks = post("/tokenize", {"content": text})["tokens"]

        if a.fill:
            # add to an existing manifest rather than replacing it; keep it sorted by depth
            manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else []
            for d in depths:
                doc = doc_at(text, toks, d)
                prefix, full = render(doc, QUESTIONS[0])
                tp = post("/tokenize", {"content": prefix, "add_special": False, "parse_special": True})["tokens"]
                tf = post("/tokenize", {"content": full, "add_special": False, "parse_special": True})["tokens"]
                if tf[:len(tp)] != tp:
                    sys.exit(f"depth {d}: the prefix does not tokenize as a prefix of the full prompt")
                t0 = time.time()
                r = post("/completion", {"prompt": prefix, "n_predict": 0, "cache_prompt": True})
                name = f"q38-{len(tp)}.bin"
                s = post(f"/slots/0?action=save", {"filename": name})
                pf = f"q38-{len(tp)}.prompt.txt"
                open(os.path.join(slot_dir, pf), "w", encoding="utf-8").write(prefix)
                # the template tail after the question, so --restore can rebuild the full prompt
                tail = full[len(prefix) + len(QUESTIONS[0]):]
                manifest = [e for e in manifest if e["file"] != name]
                manifest.append({"file": name, "prompt_file": pf, "n_tokens": len(tp), "tail": tail})
                manifest.sort(key=lambda e: e["n_tokens"])
                json.dump(manifest, open(manifest_path, "w"), indent=1)
                print(f"saved {name}: {len(tp)} tokens, {s.get('n_saved', '?')} saved, "
                      f"{s.get('n_written', 0)/2**30:.2f} GiB, fill took {time.time()-t0:.0f} s, "
                      f"gpu0 min free {min_free[0]} MiB", flush=True)
            return

        for d in depths:
            doc = post("/detokenize", {"tokens": toks[:d]})["content"]
            for qi, q in enumerate(QUESTIONS):
                t0 = time.time()
                r = post("/v1/chat/completions", {
                    "messages": [{"role": "user", "content": doc + "\n\n" + q}],
                    "max_tokens": a.n_predict, "cache_prompt": True, "seed": 1234 + qi,
                })
                rec = record(a.label, d, qi, r, t0, min_free[0])
                out.write(json.dumps(rec) + "\n"); out.flush(); show(rec)
    finally:
        stop.set()
        if srv.poll() is None:
            srv.send_signal(signal.SIGTERM)
            try:
                srv.wait(60)
            except subprocess.TimeoutExpired:
                srv.kill()


def render_tail(prefix_plus_q, ent):
    return prefix_plus_q + ent["tail"]


if __name__ == "__main__":
    main()
