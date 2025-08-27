# train_architecture.py
import os, math, json, time, argparse, random
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
from transformers import AutoTokenizer
from current_architecture import build_model

def ddp_is_on():
  return int(os.environ.get("WORLD_SIZE", "1")) > 1

def ddp_setup():
  if ddp_is_on():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def ddp_cleanup():
  if ddp_is_on() and dist.is_initialized():
    dist.destroy_process_group()

def set_seed(s):
  random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class TextSampler:
  def __init__(self, tok, ds, col, seq_len, pack_multiple=1):
    self.tok, self.ds, self.col, self.seq_len = tok, ds, col, seq_len
    self.pm = pack_multiple
  def __len__(self):
    return len(self.ds)
  def __iter__(self):
    for ex in self.ds:
      txt = ex[self.col]
      ids = self.tok(txt, truncation=False, add_special_tokens=False)["input_ids"]
      if len(ids) < self.seq_len:
        ids = ids + [self.tok.eos_token_id]*(self.seq_len - len(ids))
        yield torch.tensor(ids[:self.seq_len], dtype=torch.long)
      else:
        i = 0
        while i + self.seq_len <= len(ids):
          yield torch.tensor(ids[i:i+self.seq_len], dtype=torch.long)
          i += self.seq_len*self.pm

def collate(batch):
  x = torch.stack(batch)  # (B, T)
  return x[:, :-1], x[:, 1:]

def cycle(loader):
  while True:
    for b in loader:
      yield b

def save_ckpt(path, model, opt, sch, step, extra=None):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  torch.save({
    "model": model.state_dict(),
    "opt": opt.state_dict() if opt else None,
    "sch": sch.state_dict() if sch else None,
    "step": step,
    "extra": extra or {},
  }, path)

def load_ckpt(path, model, opt=None, sch=None, map_location="cpu"):
  ck = torch.load(path, map_location=map_location)
  model.load_state_dict(ck["model"], strict=True)
  if opt and ck["opt"]: opt.load_state_dict(ck["opt"])
  if sch and ck["sch"]: sch.load_state_dict(ck["sch"])
  return ck.get("step", 0), ck.get("extra", {})

def evaluate(model, data_iter, steps, amp_dtype=torch.bfloat16):
  model.eval()
  n, tot = 0, 0.0
  with torch.no_grad():
    for _ in range(steps):
      xb, yb = next(data_iter)
      xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
      with torch.autocast(device_type="cuda", dtype=amp_dtype):
        logits = model(xb, write_mem=False)
        loss = torch.nn.functional.cross_entropy(
          logits.reshape(-1, logits.size(-1)), yb.reshape(-1), ignore_index=-100
        )
      if ddp_is_on():
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss = loss / dist.get_world_size()
      tot += float(loss.item())
      n += 1
  model.train()
  avg = tot / max(1, n)
  return {"loss": avg, "ppl": math.exp(min(20.0, avg))}

def main():
  p = argparse.ArgumentParser()
  p.add_argument("--model_config_json", type=str, default=None)
  p.add_argument("--model_override", type=str, default=None)
  p.add_argument("--tokenizer", type=str, default="gpt2")
  p.add_argument("--dataset", type=str, default="wikitext")
  p.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
  p.add_argument("--text_column", type=str, default="text")
  p.add_argument("--seq_len", type=int, default=2048)
  p.add_argument("--batch_size", type=int, default=2)
  p.add_argument("--grad_accum", type=int, default=8)
  p.add_argument("--lr", type=float, default=2e-4)
  p.add_argument("--warmup_steps", type=int, default=100)
  p.add_argument("--max_steps", type=int, default=5000)
  p.add_argument("--eval_every", type=int, default=200)
  p.add_argument("--eval_batches", type=int, default=50)
  p.add_argument("--save_every", type=int, default=500)
  p.add_argument("--out_dir", type=str, default="runs/h1_titans")
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--fp16", action="store_true")
  p.add_argument("--bf16", action="store_true")
  p.add_argument("--compile", action="store_true")
  p.add_argument("--resume", type=str, default=None)
  p.add_argument("--grad_clip", type=float, default=1.0)
  p.add_argument("--weight_decay", type=float, default=0.1)
  p.add_argument("--adam_beta1", type=float, default=0.9)
  p.add_argument("--adam_beta2", type=float, default=0.95)
  p.add_argument("--adam_eps", type=float, default=1e-8)
  p.add_argument("--disable_mem_writes", action="store_true")
  args = p.parse_args()

  ddp_setup()
  is_master = (not ddp_is_on()) or int(os.environ.get("RANK", "0")) == 0
  set_seed(args.seed)

  tok = AutoTokenizer.from_pretrained(args.tokenizer)
  if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token if tok.eos_token else "</s>"

  if args.dataset_config and len(args.dataset_config) > 0:
    ds_tr = load_dataset(args.dataset, args.dataset_config, split="train")
    ds_ev = load_dataset(args.dataset, args.dataset_config, split="validation")
  else:
    ds_tr = load_dataset(args.dataset, split="train")
    ds_ev = load_dataset(args.dataset, split="validation")

  tr_iter = cycle(torch.utils.data.DataLoader(
    TextSampler(tok, ds_tr, args.text_column, args.seq_len),
    batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True,
    collate_fn=collate, drop_last=True,
  ))
  ev_iter = cycle(torch.utils.data.DataLoader(
    TextSampler(tok, ds_ev, args.text_column, args.seq_len),
    batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True,
    collate_fn=collate, drop_last=True,
  ))

  cfg = {}
  if args.model_config_json:
    with open(args.model_config_json, "r") as f:
      cfg = json.load(f)
  if args.model_override:
    cfg.update(json.loads(args.model_override))

  m = build_model(cfg)
  m.cuda()
  if args.compile: m = torch.compile(m, mode="reduce-overhead", fullgraph=False)

  opt = torch.optim.AdamW(
    m.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
    eps=args.adam_eps, weight_decay=args.weight_decay
  )

  def lr_lambda(step):
    if step < args.warmup_steps:
      return max(1e-8, float(step+1)/max(1, args.warmup_steps))
    return 1.0
  sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

  start_step = 0
  best_eval = float("inf")
  if args.resume:
    start_step, extra = load_ckpt(args.resume, m, opt, sch, map_location="cpu")
    best_eval = extra.get("best_eval", best_eval)

  if ddp_is_on():
    m = DDP(m, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=False, gradient_as_bucket_view=True)

  amp_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else None)
  scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

  m.train()
  step = start_step
  t0 = time.time()
  while step < args.max_steps:
    opt.zero_grad(set_to_none=True)
    losses = []
    for _ in range(args.grad_accum):
      xb, yb = next(tr_iter)
      xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
      if amp_dtype is None:
        logits = m(xb, write_mem=not args.disable_mem_writes and False)
        loss = torch.nn.functional.cross_entropy(
          logits.reshape(-1, logits.size(-1)), yb.reshape(-1), ignore_index=-100
        ) / args.grad_accum
        loss.backward()
      else:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
          logits = m(xb, write_mem=not args.disable_mem_writes and False)
          loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), yb.reshape(-1), ignore_index=-100
          ) / args.grad_accum
        if args.fp16:
          scaler.scale(loss).backward()
        else:
          loss.backward()
      losses.append(float(loss.item())*args.grad_accum)

    if args.grad_clip:
      if args.fp16:
        scaler.unscale_(opt)
      torch.nn.utils.clip_grad_norm_(m.parameters(), args.grad_clip)

    if args.fp16:
      scaler.step(opt); scaler.update()
    else:
      opt.step()
    sch.step()
    step += 1

    if is_master and step % 10 == 0:
      dt = time.time() - t0; t0 = time.time()
      tr_loss = sum(losses)/len(losses)
      lr = sch.get_last_lr()[0]
      print(f"step {step} | loss {tr_loss:.4f} | lr {lr:.6g} | dt {dt:.2f}s")

    if step % args.eval_every == 0:
      if ddp_is_on(): dist.barrier()
      if is_master:
        metrics = evaluate(m.module if ddp_is_on() else m, ev_iter, args.eval_batches,
                           amp_dtype=(torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32))
        print(f"[eval] step {step} | loss {metrics['loss']:.4f} | ppl {metrics['ppl']:.2f}")
        if metrics["loss"] < best_eval:
          best_eval = metrics["loss"]
          save_ckpt(os.path.join(args.out_dir, "best.pt"),
                    m.module if ddp_is_on() else m, opt, sch, step,
                    extra={"best_eval": best_eval})
      if ddp_is_on(): dist.barrier()

    if step % args.save_every == 0 and is_master:
      save_ckpt(os.path.join(args.out_dir, f"ckpt_{step}.pt"),
                m.module if ddp_is_on() else m, opt, sch, step,
                extra={"best_eval": best_eval})

  if is_master:
    save_ckpt(os.path.join(args.out_dir, "last.pt"),
              m.module if ddp_is_on() else m, opt, sch, step,
              extra={"best_eval": best_eval})
  ddp_cleanup()

if __name__ == "__main__":
  main()